#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""简单的 Temporal SARL 训练示例。

该脚本展示如何读取 ICEWS 历史数据，构建训练样本，并应用 TemporalSARL。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from sarl_model import TemporalSARL


class TemporalDataset(Dataset):
    """加载 (h, r, t, ts) 训练样本。"""

    def __init__(self, triples: List[Tuple[int, int, int, float]], max_history: int = 4):
        self.triples = triples
        self.max_history = max_history

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int):
        head, rel, tail, ts = self.triples[idx]
        history_entities = [head] * self.max_history
        history_times = [0.0] * self.max_history
        return {
            "history_entities": torch.tensor(history_entities, dtype=torch.long),
            "history_times": torch.tensor(history_times, dtype=torch.float),
            "query_relation": torch.tensor(rel, dtype=torch.long),
            "positive_tail": torch.tensor(tail, dtype=torch.long),
            "timestamp": torch.tensor(ts, dtype=torch.float),
        }


def load_triples(edge_csv: Path, limit: int = 10000) -> List[Tuple[int, int, int, float]]:
    triples: List[Tuple[int, int, int, float]] = []
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
            if len(triples) >= limit:
                break
    return triples


def train(args: argparse.Namespace) -> None:
    edge_file = Path(args.edge_file)
    triples = load_triples(edge_file, limit=args.sample_limit)
    dataset = TemporalDataset(triples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = TemporalSARL(
        num_entities=args.num_entities,
        num_relations=args.num_relations,
        embed_dim=args.embed_dim,
    )
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            scores = model(
                batch["history_entities"],
                batch["history_times"],
                batch["query_relation"],
                batch["positive_tail"].unsqueeze(1),
                batch["query_relation"].unsqueeze(1),
                batch["timestamp"].unsqueeze(1),
            )
            loss = F.binary_cross_entropy_with_logits(scores, torch.ones_like(scores))
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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--sample_limit", type=int, default=10000)
    parser.add_argument(
        "--save_path", type=str, default="./sarl_model.pth", help="模型权重保存路径"
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
