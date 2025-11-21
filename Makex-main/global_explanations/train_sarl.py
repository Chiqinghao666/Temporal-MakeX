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
        # 这里简化处理，历史路径暂时填充为当前节点本身(Self-loop)
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
    # 1. 自动检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    edge_file = Path(args.edge_file)
    triples = load_triples(edge_file, limit=args.sample_limit)
    dataset = TemporalDataset(triples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = TemporalSARL(
        num_entities=args.num_entities,
        num_relations=args.num_relations,
        embed_dim=args.embed_dim,
    )

    # 2. 将模型搬到 GPU
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            # 3. 将数据搬到 GPU
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # 正样本前向传播
            pos_scores = model(
                batch["history_entities"],
                batch["history_times"],
                batch["query_relation"],
                batch["positive_tail"].unsqueeze(1),
                batch["query_relation"].unsqueeze(1),
                batch["timestamp"].unsqueeze(1),
            )
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_scores, torch.ones_like(pos_scores)
            )

            # 负采样逻辑 (确保生成的随机 Tensor 也在正确的 device 上)
            neg_tail = torch.randint(
                low=0,
                high=args.num_entities,
                size=batch["positive_tail"].shape,
                dtype=torch.long,
                device=device  # 关键：指定设备
            )

            # 简单去重：确保负样本不等于正样本
            mask = neg_tail.eq(batch["positive_tail"])
            while mask.any():
                neg_tail[mask] = torch.randint(
                    0, args.num_entities, size=(mask.sum().item(),),
                    dtype=torch.long, device=device
                )
                mask = neg_tail.eq(batch["positive_tail"])

            # 负样本前向传播
            neg_scores = model(
                batch["history_entities"],
                batch["history_times"],
                batch["query_relation"],
                neg_tail.unsqueeze(1),
                batch["query_relation"].unsqueeze(1),
                batch["timestamp"].unsqueeze(1),
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_scores, torch.zeros_like(neg_scores)
            )

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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--sample_limit", type=int, default=10000)
    parser.add_argument(
        "--save_path", type=str, default="./sarl_model.pth", help="模型权重保存路径"
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())