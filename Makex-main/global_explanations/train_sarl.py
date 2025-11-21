# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""简单的 Temporal SARL 训练示例 (修复维度版)。"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from sarl_model import TemporalSARL

HistoryRecord = Tuple[int, int, float]  # tail, relation, timestamp
TripleRecord = Tuple[int, int, int, float]  # head, relation, tail, timestamp


def build_graph_index(triples: List[TripleRecord]) -> Dict[int, List[HistoryRecord]]:
    """Construct adjacency lists sorted by timestamp descending."""
    graph: Dict[int, List[HistoryRecord]] = defaultdict(list)
    for head, relation, tail, ts in triples:
        graph[head].append((tail, relation, ts))
    for history in graph.values():
        history.sort(key=lambda x: x[2], reverse=True)
    return graph


class TemporalDataset(Dataset):
    """Dataset that samples real temporal histories before each event."""

    def __init__(
            self,
            triples: List[TripleRecord],
            graph_index: Dict[int, List[HistoryRecord]],
            num_entities: int,
            max_history: int = 4,
            pad_id: int | None = None,
    ) -> None:
        self.triples = triples
        self.graph_index = graph_index
        self.num_entities = num_entities
        self.max_history = max_history
        self.pad_id = num_entities if pad_id is None else pad_id

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

    def _collect_history(self, head: int, timestamp: float) -> Tuple[List[int], List[float]]:
        records = self.graph_index.get(head, [])
        filtered = [rec for rec in records if rec[2] < timestamp]
        history_entities: List[int] = []
        history_times: List[float] = []
        for tail, relation, ts in filtered[: self.max_history]:
            history_entities.append(tail)
            history_times.append(max(0.0, timestamp - ts))
        pad_length = self.max_history - len(history_entities)
        if pad_length > 0:
            history_entities.extend([self.pad_id] * pad_length)
            history_times.extend([0.0] * pad_length)
        return history_entities, history_times


def load_triples(edge_csv: Path, limit: int | None = None) -> List[TripleRecord]:
    triples: List[TripleRecord] = []
    with edge_csv.open() as f:
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


def negative_sample(tails: torch.Tensor, num_entities: int) -> torch.Tensor:
    # 修改点 1: 添加 device=tails.device
    neg_tails = torch.randint(
        0, num_entities, size=tails.shape,
        dtype=torch.long, device=tails.device
    )

    mask = neg_tails.eq(tails)
    while mask.any():
        # 修改点 2: 添加 device=tails.device
        neg_tails[mask] = torch.randint(
            0, num_entities, size=(mask.sum().item(),),
            dtype=torch.long, device=tails.device
        )
        mask = neg_tails.eq(tails)
    return neg_tails


def train(args: argparse.Namespace) -> None:
    # 1. GPU 检测
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    triples = load_triples(Path(args.edge_file), limit=args.sample_limit or None)
    if not triples:
        raise RuntimeError("No triples loaded; check edge_file or sample_limit.")

    graph_index = build_graph_index(triples)

    dataset = TemporalDataset(
        triples=triples,
        graph_index=graph_index,
        num_entities=args.num_entities,
        max_history=args.max_history,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = TemporalSARL(
        num_entities=args.num_entities + 1,  # reserve PAD id
        num_relations=args.num_relations,
        embed_dim=args.embed_dim,
    )
    model.to(device)

    # 学习率调优建议：从 1e-3 开始
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            history_entities = batch["history_entities"].to(device)
            history_times = batch["history_times"].to(device)
            query_relation = batch["query_relation"].to(device)
            positive_tail = batch["positive_tail"].to(device)
            timestamps = batch["timestamp"].to(device)

            # === 关键修改：去掉了 .unsqueeze(1) ===
            pos_scores = model(
                history_entities,  # (B, HistLen)
                history_times,  # (B, HistLen)
                query_relation,
                positive_tail.unsqueeze(1),
                query_relation.unsqueeze(1),
                timestamps.unsqueeze(1),
            )
            pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))

            neg_tail = negative_sample(positive_tail, args.num_entities).to(device)

            # === 关键修改：去掉了 .unsqueeze(1) ===
            neg_scores = model(
                history_entities,  # (B, HistLen)
                history_times,  # (B, HistLen)
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
