#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Temporal SARL miner with detailed logging."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import torch

CURRENT_DIR = Path(__file__).resolve().parent
import sys

if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from sarl_model import TemporalSARL

try:
    import pyMakex
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("pyMakex module is required for SARL miner to run.") from exc


@dataclass(frozen=True)
class TemporalNeighbor:
    dst: int
    relation: int
    timestamp: float


@dataclass
class SARLOptions:
    max_hops: int = 3
    beam_size: int = 8
    min_timestamp: float = 0.0
    time_decay: float = 86400.0  # 1 day window by default
    device: str = "cpu"
    log_dir: Path = Path("global_explanations")


class SARLMiner:
    """执行 SARL 路径挖掘并输出详尽日志。"""

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
        self.log_dir = options.log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.raw_path_file = self.log_dir / "sarl_raw_paths.txt"
        self.entity_map = self._load_map(entity2id_path)
        self.relation_map = self._load_map(relation2id_path)
        self.entity_inv = {v: k for k, v in self.entity_map.items()}
        self.relation_inv = {v: k for k, v in self.relation_map.items()}

    def _load_map(self, path: Path) -> Dict[str, int]:
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    def _id_to_name(self, inv_map: Dict[int, str], idx: int) -> str:
        return inv_map.get(idx, f"ID_{idx}")

    def mine_paths(
        self,
        head_id: int,
        relation_id: int,
        query_time: float,
        num_walks: int = 100,
    ) -> List[List[TemporalNeighbor]]:
        """Main loop to mine temporal paths from head."""
        statistics = {"attempted": 0, "valid": 0}
        mined_paths: List[List[TemporalNeighbor]] = []
        with self.raw_path_file.open("a", encoding="utf-8") as fp:
            for walk_idx in range(num_walks):
                statistics["attempted"] += 1
                path = self._single_walk(head_id, relation_id, query_time, fp, walk_idx)
                if not path:
                    continue
                statistics["valid"] += 1
                mined_paths.append(path)
        attempted = max(statistics["attempted"], 1)
        print(
            "[SARL Performance]\n"
            f"- Total Walks Attempted: {statistics['attempted']}\n"
            f"- Valid Paths Found (Hits): {statistics['valid']}\n"
            f"- Hit Rate: {statistics['valid'] / attempted * 100:.2f}%"
        )
        return mined_paths

    def _single_walk(
        self,
        head_id: int,
        relation_id: int,
        query_time: float,
        log_fp,
        walk_idx: int,
    ) -> List[TemporalNeighbor]:
        current = head_id
        current_time = query_time
        history_entities = [current]
        history_times = [0.0]
        mined_path: List[TemporalNeighbor] = []
        for hop in range(self.options.max_hops):
            neighbors = self._temporal_neighbors(current, current_time)
            if not neighbors:
                print(f"[SARL Step] Walk {walk_idx}, hop {hop}: no neighbors, terminate.")
                return []
            cand_entities = torch.tensor([[n.dst for n in neighbors]], dtype=torch.long)
            cand_relations = torch.tensor([[n.relation for n in neighbors]], dtype=torch.long)
            time_diff = torch.tensor(
                [[max(0.0, current_time - n.timestamp) for n in neighbors]],
                dtype=torch.float,
            )
            hist_entity_tensor = torch.tensor([history_entities], dtype=torch.long)
            hist_time_tensor = torch.tensor([history_times], dtype=torch.float)
            relation_tensor = torch.tensor([relation_id], dtype=torch.long)

            self.model.eval()
            with torch.no_grad():
                scores = self.model(
                    hist_entity_tensor.to(self.options.device),
                    hist_time_tensor.to(self.options.device),
                    relation_tensor.to(self.options.device),
                    cand_entities.to(self.options.device),
                    cand_relations.to(self.options.device),
                    time_diff.to(self.options.device),
                )
            probs = torch.softmax(scores.squeeze(0), dim=-1)
            top_prob, top_idx = torch.max(probs, dim=-1)
            selected = neighbors[top_idx.item()]
            print(
                f"[SARL Step] Walk {walk_idx}, hop {hop}\n"
                f"  Current: {self._id_to_name(self.entity_inv, current)} "
                f"(time={self._format_ts(current_time)})\n"
                f"  [Model Decision] top prob {top_prob.item():.3f} -> "
                f"{self._id_to_name(self.relation_inv, selected.relation)} "
                f"@ {self._format_ts(selected.timestamp)}"
            )
            mined_path.append(selected)
            log_fp.write(
                f"Query({self._id_to_name(self.entity_inv, head_id)}, "
                f"{self._id_to_name(self.relation_inv, relation_id)}) -> "
                f"Step{hop}: {self._format_edge(current, selected)}\n"
            )
            current = selected.dst
            history_entities.append(current)
            history_times.append(max(0.0, query_time - selected.timestamp))
            current_time = selected.timestamp
        return mined_path

    def _temporal_neighbors(self, node_id: int, time_upper: float) -> List[TemporalNeighbor]:
        ts_upper = int(time_upper)
        ts_lower = int(max(self.options.min_timestamp, time_upper - self.options.time_decay))
        neighbor_ids: Sequence[int] = []
        try:
            neighbor_ids = pyMakex.GetTemporalNeighbors(
                self.graph_ptr, int(node_id), ts_lower, ts_upper, 1
            )
        except TypeError:
            # 兼容旧的 pyMakex 版本
            neighbor_ids = pyMakex.GetTemporalNeighbors(int(node_id), ts_lower, ts_upper)

        edges: List[TemporalNeighbor] = []
        candidates = set(neighbor_ids) if neighbor_ids else None
        for edge in self.edge_store.get(node_id, []):
            if candidates is not None and edge.dst not in candidates:
                continue
            if edge.timestamp > ts_upper or edge.timestamp < ts_lower:
                continue
            edges.append(edge)
        if not edges:
            for edge in self.edge_store.get(node_id, []):
                if ts_lower <= edge.timestamp <= ts_upper:
                    edges.append(edge)
        edges.sort(key=lambda n: n.timestamp, reverse=True)
        if len(edges) > self.options.beam_size:
            edges = edges[: self.options.beam_size]
        return edges

    def _format_edge(self, src: int, edge: TemporalNeighbor) -> str:
        src_name = self._id_to_name(self.entity_inv, src)
        rel_name = self._id_to_name(self.relation_inv, edge.relation)
        dst_name = self._id_to_name(self.entity_inv, edge.dst)
        return f"{src_name} --[{rel_name}, {self._format_ts(edge.timestamp)}]--> {dst_name}"

    @staticmethod
    def _format_ts(ts: float) -> str:
        try:
            return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
        except (OverflowError, OSError, ValueError):
            return f"{ts:.2f}"


__all__ = ["SARLMiner", "SARLOptions", "TemporalNeighbor"]
