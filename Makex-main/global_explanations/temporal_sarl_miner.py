#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Temporal SARL miner with Transformer policy and rich logging."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from sarl_model import TemporalSARL

try:
    import pyMakex  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("pyMakex module is required for SARL miner.") from exc


@dataclass(frozen=True)
class TemporalNeighbor:
    dst: int
    relation: int
    timestamp: float


@dataclass
class TemporalPath:
    head: int
    relation: int
    query_time: float
    edges: List[TemporalNeighbor]


@dataclass
class SARLOptions:
    max_hops: int = 3
    history_size: int = 5
    beam_size: int = 8
    time_window: float = 30 * 86400.0
    min_timestamp: float = 0.0
    device: str = "cpu"
    log_dir: Path = Path(".")


class SARLMiner:
    """Run SARL walks on pyMakex graphs with time constraints."""

    def __init__(
        self,
        model: TemporalSARL,
        options: SARLOptions,
        entity2id_path: Path,
        relation2id_path: Path,
        graph_ptr: int,
        edge_store: Dict[int, List[TemporalNeighbor]],
    ) -> None:
        self.model = model.to(options.device)
        self.options = options
        self.graph_ptr = graph_ptr
        self.edge_store = edge_store
        self.entity_map = self._load_map(entity2id_path)
        self.relation_map = self._load_map(relation2id_path)
        self.entity_inv = {v: k for k, v in self.entity_map.items()}
        self.relation_inv = {v: k for k, v in self.relation_map.items()}
        self.pad_entity = len(self.entity_map)
        self.pad_relation = len(self.relation_map)
        self.log_dir = options.log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.raw_path_file = self.log_dir / "sarl_raw_paths.txt"
        self.raw_path_file.write_text("")
        self.reset_statistics()

    def reset_statistics(self) -> None:
        self.stats_attempted = 0
        self.stats_hits = 0

    @staticmethod
    def _load_map(path: Path) -> Dict[str, int]:
        if not path.exists():
            raise FileNotFoundError(f"Mapping file not found: {path}")
        return json.loads(path.read_text())

    def _id_to_name(self, inv_map: Dict[int, str], idx: int) -> str:
        return inv_map.get(idx, f"ID_{idx}")

    def mine_paths(
        self,
        head_id: int,
        relation_id: int,
        query_time: float,
        num_walks: int,
    ) -> List[TemporalPath]:
        results: List[TemporalPath] = []
        for walk_idx in range(num_walks):
            self.stats_attempted += 1
            path = self._single_walk(head_id, relation_id, query_time, walk_idx)
            if path:
                self.stats_hits += 1
                results.append(path)
        return results

    def _single_walk(
        self,
        head_id: int,
        relation_id: int,
        query_time: float,
        walk_idx: int,
    ) -> Optional[TemporalPath]:
        history_entities, history_relations, history_deltas = self._init_history(head_id, relation_id)
        current = head_id
        current_time = query_time
        mined_edges: List[TemporalNeighbor] = []
        for hop in range(self.options.max_hops):
            neighbors = self._temporal_neighbors(current, current_time)
            if not neighbors:
                print(f"[SARL Step] Walk {walk_idx}, hop {hop}: no neighbors in window.")
                return None
            choice, probs = self._select_neighbor(
                neighbors,
                history_entities,
                history_relations,
                history_deltas,
                current,
                relation_id,
                query_time,
            )
            mined_edges.append(choice)
            self._log_step(
                walk_idx,
                hop,
                current,
                relation_id,
                current_time,
                neighbors,
                probs,
                choice,
            )
            self._append_history(
                history_entities,
                history_relations,
                history_deltas,
                choice.dst,
                choice.relation,
                max(0.0, query_time - choice.timestamp),
            )
            current = choice.dst
            current_time = choice.timestamp
            if len(mined_edges) == self.options.max_hops:
                break
        if mined_edges:
            self._write_raw_path(head_id, relation_id, query_time, mined_edges)
            return TemporalPath(head=head_id, relation=relation_id, query_time=query_time, edges=mined_edges)
        return None

    def _init_history(self, head: int, relation: int) -> Tuple[List[int], List[int], List[float]]:
        entities = [head]
        relations = [relation]
        deltas = [0.0]
        while len(entities) < self.options.history_size:
            entities.append(self.pad_entity)
            relations.append(self.pad_relation)
            deltas.append(0.0)
        return entities, relations, deltas

    def _append_history(
        self,
        entities: List[int],
        relations: List[int],
        deltas: List[float],
        entity: int,
        relation: int,
        delta: float,
    ) -> None:
        entities.append(entity)
        relations.append(relation)
        deltas.append(delta)
        if len(entities) > self.options.history_size:
            entities.pop(0)
            relations.pop(0)
            deltas.pop(0)

    def _history_tensors(
        self,
        entities: List[int],
        relations: List[int],
        deltas: List[float],
        current_entity: int,
        relation_id: int,
    ) -> Tuple[torch.Tensor, ...]:
        device = self.options.device
        hist_entities = torch.tensor([entities], dtype=torch.long, device=device)
        hist_relations = torch.tensor([relations], dtype=torch.long, device=device)
        hist_deltas = torch.tensor([deltas], dtype=torch.float32, device=device)
        current_tensor = torch.tensor([current_entity], dtype=torch.long, device=device)
        relation_tensor = torch.tensor([relation_id], dtype=torch.long, device=device)
        return hist_entities, hist_relations, hist_deltas, current_tensor, relation_tensor

    def _select_neighbor(
        self,
        neighbors: List[TemporalNeighbor],
        history_entities: List[int],
        history_relations: List[int],
        history_deltas: List[float],
        current_entity: int,
        relation_id: int,
        query_time: float,
    ) -> Tuple[TemporalNeighbor, torch.Tensor]:
        cand_entities = torch.tensor([[n.dst for n in neighbors]], dtype=torch.long, device=self.options.device)
        cand_relations = torch.tensor([[n.relation for n in neighbors]], dtype=torch.long, device=self.options.device)
        cand_deltas = torch.tensor(
            [[max(0.0, query_time - n.timestamp) for n in neighbors]],
            dtype=torch.float32,
            device=self.options.device,
        )
        hist_entities, hist_relations, hist_deltas, current_tensor, relation_tensor = self._history_tensors(
            history_entities,
            history_relations,
            history_deltas,
            current_entity,
            relation_id,
        )
        self.model.eval()
        with torch.no_grad():
            scores = self.model(
                hist_entities,
                hist_relations,
                hist_deltas,
                current_tensor,
                relation_tensor,
                cand_entities,
                cand_relations,
                cand_deltas,
            )
        probs = torch.softmax(scores.squeeze(0), dim=-1)
        top_idx = torch.argmax(probs).item()
        return neighbors[top_idx], probs

    def _temporal_neighbors(self, node_id: int, time_upper: float) -> List[TemporalNeighbor]:
        ts_upper = int(time_upper)
        ts_lower = int(max(self.options.min_timestamp, time_upper - self.options.time_window))
        candidate_ids: Optional[Iterable[int]] = None
        try:
            candidate_ids = pyMakex.GetTemporalNeighbors(
                self.graph_ptr, int(node_id), ts_lower, ts_upper, 1
            )
        except TypeError:
            candidate_ids = pyMakex.GetTemporalNeighbors(int(node_id), ts_lower, ts_upper)

        neighbors: List[TemporalNeighbor] = []
        allowed = set(candidate_ids) if candidate_ids else None
        for edge in self.edge_store.get(node_id, []):
            if not (ts_lower <= edge.timestamp <= ts_upper):
                continue
            if allowed is not None and edge.dst not in allowed:
                continue
            neighbors.append(edge)
        neighbors.sort(key=lambda e: e.timestamp, reverse=True)
        if len(neighbors) > self.options.beam_size:
            neighbors = neighbors[: self.options.beam_size]
        return neighbors

    def _log_step(
        self,
        walk_idx: int,
        hop: int,
        current: int,
        goal_relation: int,
        current_time: float,
        neighbors: List[TemporalNeighbor],
        probs: torch.Tensor,
        selected: TemporalNeighbor,
    ) -> None:
        entity_name = self._id_to_name(self.entity_inv, current)
        goal_name = self._id_to_name(self.relation_inv, goal_relation)
        current_time_str = self._format_ts(current_time)
        print(f"[SARL Step] Walk {walk_idx}, hop {hop}, Current: {entity_name} (t={current_time_str}), Goal Rel: {goal_name}")
        sorted_idx = torch.argsort(probs, descending=True)
        top = sorted_idx[: min(3, len(sorted_idx))]
        top_desc = ", ".join(
            f"{self._id_to_name(self.relation_inv, neighbors[i].relation)}({probs[i].item():.2f})" for i in top
        )
        low_idx = sorted_idx[-1].item()
        low_desc = f"{self._id_to_name(self.relation_inv, neighbors[low_idx].relation)}({probs[low_idx].item():.2f})"
        print(f"  [Model Decision] High scores for: {top_desc}")
        print(f"  [Model Decision] Low scores for: {low_desc}")
        print(
            f"  [Action] Selected {self._id_to_name(self.relation_inv, selected.relation)}"
            f" -> {self._id_to_name(self.entity_inv, selected.dst)}"
            f" at {self._format_ts(selected.timestamp)}"
        )

    def _write_raw_path(
        self,
        head: int,
        relation: int,
        query_time: float,
        edges: List[TemporalNeighbor],
    ) -> None:
        human_edges = " -> ".join(
            f"{self._id_to_name(self.relation_inv, edge.relation)}"
            f"({self._format_ts(edge.timestamp)}) => {self._id_to_name(self.entity_inv, edge.dst)}"
            for edge in edges
        )
        with self.raw_path_file.open("a", encoding="utf-8") as fp:
            fp.write(
                f"Query({self._id_to_name(self.entity_inv, head)}, {self._id_to_name(self.relation_inv, relation)}"
                f" @ {self._format_ts(query_time)}) -> Path: {human_edges}\n"
            )

    def report_performance(self) -> None:
        attempted = max(1, self.stats_attempted)
        hit_rate = self.stats_hits / attempted * 100
        print(
            "[SARL Performance]\n"
            f"- Total Walks Attempted: {self.stats_attempted}\n"
            f"- Valid Paths Found (Hits): {self.stats_hits}\n"
            f"- Hit Rate: {hit_rate:.2f}% (Benchmark: Random Walk < 1%)"
        )

    def cluster_paths(
        self, paths: List[TemporalPath], time_bucket: float
    ) -> Dict[Tuple[int, int, str], List[TemporalPath]]:
        grouped: Dict[Tuple[int, int, str], List[TemporalPath]] = {}
        for path in paths:
            signature = self._build_signature(path, time_bucket)
            key = (path.head, path.relation, signature)
            grouped.setdefault(key, []).append(path)
        return grouped

    def _build_signature(self, path: TemporalPath, bucket: float) -> str:
        tokens = [f"L{len(path.edges)}"]
        for edge in path.edges:
            delta = max(0.0, path.query_time - edge.timestamp)
            tokens.append(f"{edge.relation}:{int(delta // bucket)}")
        return "|".join(tokens)

    def path_to_rep(self, path: TemporalPath, support: int) -> List:
        node_ids = {path.head: 1}
        next_idx = 2
        current = path.head
        edges: List[List[int]] = []
        for edge in path.edges:
            if edge.dst not in node_ids:
                node_ids[edge.dst] = next_idx
                next_idx += 1
            src_idx = node_ids[current]
            dst_idx = node_ids[edge.dst]
            edges.append([src_idx, dst_idx, edge.relation])
            current = edge.dst
        vertices = [[idx, 0] for _, idx in sorted(node_ids.items(), key=lambda item: item[1])]
        predicates = [
            ["Constant", 1, "query_relation", str(path.relation), "string", "="],
            ["Constant", 1, "path_length", str(len(path.edges)), "string", "="],
        ]
        stats = [float(support), 1.0]
        meta = [1, 2, 1, 1.0]
        return [vertices, edges, predicates, stats, meta]

    @staticmethod
    def _format_ts(ts: float) -> str:
        try:
            return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
        except (ValueError, OSError, OverflowError):
            return f"{ts:.2f}"


__all__ = [
    "SARLMiner",
    "SARLOptions",
    "TemporalNeighbor",
    "TemporalPath",
]
