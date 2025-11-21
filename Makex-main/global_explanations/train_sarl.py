#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Temporal SARL training script with real historical context and negative sampling."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from sarl_model import TemporalSARL

TripleRecord = Tuple[int, int, int, float]
HistoryRecord = Tuple[int, int, float]


def load_mapping(path: Path) -> Dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(f"Mapping file not found: {path}")
    return json.loads(path.read_text())


def infer_dataset_dir(edge_file: Path) -> Path:
    for parent in edge_file.resolve().parents:
        if (parent / "train.txt").exists() and (parent / "entity2id.json").exists():
            return parent
    raise RuntimeError("Unable to infer dataset directory; please provide --dataset_dir.")


def parse_temporal_line(
    line: str,
    entity_map: Dict[str, int],
    relation_map: Dict[str, int],
) -> Optional[TripleRecord]:
    parts = line.strip().split("\t")
    if len(parts) < 4:
        return None
    head, relation, tail, date_str = parts[:4]
    if head not in entity_map or tail not in entity_map or relation not in relation_map:
        return None
    try:
        ts = datetime.strptime(date_str, "%Y-%m-%d").timestamp()
    except ValueError:
        return None
    return (entity_map[head], relation_map[relation], entity_map[tail], ts)


def load_temporal_triples(
    dataset_dir: Path,
    entity_map: Dict[str, int],
    relation_map: Dict[str, int],
    limit: Optional[int] = None,
) -> List[TripleRecord]:
    triples: List[TripleRecord] = []
    for split in ("train.txt", "valid.txt", "test.txt"):
        path = dataset_dir / split
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                parsed = parse_temporal_line(line, entity_map, relation_map)
                if not parsed:
                    continue
                triples.append(parsed)
                if limit and len(triples) >= limit:
                    return triples
    return triples


def fallback_load_from_csv(edge_file: Path, limit: Optional[int]) -> List[TripleRecord]:
    triples: List[TripleRecord] = []
    with edge_file.open() as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            ts = float(row.get("timestamp:float", idx))
            triples.append(
                (
                    int(row["source_id:int"]),
                    int(row["label_id:int"]),
                    int(row["target_id:int"]),
                    ts,
                )
            )
            if limit and len(triples) >= limit:
                break
    return triples


def build_graph_index(triples: List[TripleRecord]) -> Dict[int, List[HistoryRecord]]:
    graph: Dict[int, List[HistoryRecord]] = defaultdict(list)
    for head, relation, tail, ts in triples:
        graph[head].append((tail, relation, ts))
    for history in graph.values():
        history.sort(key=lambda rec: rec[2], reverse=True)
    return graph


class TemporalDataset(Dataset):
    def __init__(
        self,
        triples: List[TripleRecord],
        graph_index: Dict[int, List[HistoryRecord]],
        num_entities: int,
        max_history: int = 4,
    ) -> None:
        self.triples = triples
        self.graph_index = graph_index
        self.max_history = max_history
        self.pad_id = num_entities  # use additional embedding slot for PAD

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int):
        head, relation, tail, timestamp = self.triples[idx]
        history_entities, history_times = self._collect_history(head, timestamp)
        return {
            "history_entities": torch.tensor(history_entities, dtype=torch.long),
            "history_times": torch.tensor(history_times, dtype=torch.float),
            "query_relation": torch.tensor(relation, dtype=torch.long),
            "positive_tail": torch.tensor(tail, dtype=torch.long),
            "timestamp": torch.tensor(timestamp, dtype=torch.float),
        }

    def _collect_history(self, head: int, current_ts: float) -> Tuple[List[int], List[float]]:
        records = self.graph_index.get(head, [])
        past = [rec for rec in records if rec[2] < current_ts]
        history_entities: List[int] = []
        history_times: List[float] = []
        for tail, _, ts in past[: self.max_history]:
            history_entities.append(tail)
            history_times.append(max(0.0, current_ts - ts))
        pad_len = self.max_history - len(history_entities)
        if pad_len > 0:
            history_entities.extend([self.pad_id] * pad_len)
            history_times.extend([0.0] * pad_len)
        return history_entities, history_times


def negative_sample(tails: torch.Tensor, num_entities: int) -> torch.Tensor:
    neg_tails = torch.randint(0, num_entities, size=tails.shape, dtype=torch.long)
    mask = neg_tails.eq(tails)
    while mask.any():
        neg_tails[mask] = torch.randint(0, num_entities, size=(mask.sum().item(),), dtype=torch.long)
        mask = neg_tails.eq(tails)
    return neg_tails


def train(args: argparse.Namespace) -> None:
    edge_file = Path(args.edge_file)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else infer_dataset_dir(edge_file)
    entity_map = load_mapping(dataset_dir / "entity2id.json")
    relation_map = load_mapping(dataset_dir / "relation2id.json")

    triples = load_temporal_triples(dataset_dir, entity_map, relation_map, args.sample_limit or None)
    if not triples:
        print("Warning: temporal splits not found, falling back to edge CSV without timestamps.")
        triples = fallback_load_from_csv(edge_file, args.sample_limit or None)
    if not triples:
        raise RuntimeError("No triples available for training.")

    graph_index = build_graph_index(triples)
    dataset = TemporalDataset(
        triples=triples,
        graph_index=graph_index,
        num_entities=args.num_entities,
        max_history=args.max_history,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = TemporalSARL(
        num_entities=args.num_entities + 1,
        num_relations=args.num_relations,
        embed_dim=args.embed_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            history_entities = batch["history_entities"].to(device)
            history_times = batch["history_times"].to(device)
            query_relation = batch["query_relation"].to(device)
            positive_tail = batch["positive_tail"].to(device)
            timestamps = batch["timestamp"].to(device)

            pos_scores = model(
                history_entities,
                history_times,
                query_relation,
                positive_tail.unsqueeze(1),
                query_relation.unsqueeze(1),
                timestamps.unsqueeze(1),
            )
            pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))

            neg_tail = negative_sample(positive_tail, args.num_entities).to(device)
            neg_scores = model(
                history_entities,
                history_times,
                query_relation,
                neg_tail.unsqueeze(1),
                query_relation.unsqueeze(1),
                timestamps.unsqueeze(1),
            )
            neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))

            loss = pos_loss + neg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[Epoch {epoch}] loss={avg_loss:.4f}")

    torch.save(model.state_dict(), args.save_path)
    print(f"[Checkpoint] Saved SARL model to {args.save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Temporal SARL model.")
    parser.add_argument("--edge_file", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, default="")
    parser.add_argument("--num_entities", type=int, required=True)
    parser.add_argument("--num_relations", type=int, required=True)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_history", type=int, default=4)
    parser.add_argument("--sample_limit", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="./sarl_model.pth")
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
