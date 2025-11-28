#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training script for Temporal-SARL.
Temporal-SARL 训练脚本。
功能：
- 构建时序图索引（Graph Index）
- 为每个查询采样真实的历史上下文（保证时间单调性）
- 应用负采样策略（Negative Sampling）
- 支持 GPU 训练并保存模型检查点
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sarl_model import TemporalSARL

# 定义类型别名，方便阅读
Triple = Tuple[int, int, int, float]  # (head, relation, tail, timestamp)
History = Tuple[int, int, float]  # (tail, relation, timestamp)


def load_mapping(path: Path) -> Dict[str, int]:
    """
    加载 ID 映射文件（JSON格式）。
    例如：把 "China" 映射为 105，"Visit" 映射为 23。
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing mapping file: {path}")
    return json.loads(path.read_text())


def infer_dataset_dir(edge_file: Path) -> Path:
    """
    根据边文件的路径，自动推断数据集的根目录。
    寻找包含 train.txt 和 entity2id.json 的父级目录。
    """
    for parent in edge_file.resolve().parents:
        if (parent / "train.txt").exists() and (parent / "entity2id.json").exists():
            return parent
    raise RuntimeError("Unable to infer dataset directory; set --dataset_dir explicitly.")


def parse_temporal_line(
        line: str,
        entity_map: Dict[str, int],
        relation_map: Dict[str, int],
) -> Optional[Triple]:
    """
    解析单行文本数据。
    输入格式示例："China\tVisit\tJapan\t2014-01-01"
    输出：(105, 23, 208, 1388534400.0)
    """
    parts = line.strip().split("\t")
    if len(parts) < 4:
        return None
    head, relation, tail, date_str = parts[:4]

    # 检查实体和关系是否都在字典里，不在则跳过
    if head not in entity_map or tail not in entity_map or relation not in relation_map:
        return None
    try:
        # 将日期字符串转为秒级时间戳 (float)
        ts = datetime.strptime(date_str, "%Y-%m-%d").timestamp()
    except ValueError:
        return None
    return (entity_map[head], relation_map[relation], entity_map[tail], ts)


def load_temporal_triples(
        dataset_dir: Path,
        entity_map: Dict[str, int],
        relation_map: Dict[str, int],
        limit: Optional[int] = None,
) -> List[Triple]:
    """
    从 train.txt, valid.txt, test.txt 中加载所有三元组。
    这是主要的数据加载函数。
    """
    triples: List[Triple] = []
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
                # 如果设置了采样限制，达到数量即停止
                if limit and len(triples) >= limit:
                    return triples
    return triples


def fallback_load_from_csv(edge_file: Path, limit: Optional[int]) -> List[Triple]:
    """
    备用加载方案：如果txt文件不存在，尝试直接读取原始 CSV 文件。
    通常用于处理未完全预处理的数据。
    """
    triples: List[Triple] = []
    with edge_file.open() as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            # 尝试获取时间戳，如果没有则用行号代替（容错）
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


def build_graph_index(triples: List[Triple]) -> Dict[int, List[History]]:
    """
    构建历史索引（档案库）。
    Key: Head Entity ID
    Value: 该实体参与的所有历史事件列表 [(tail, rel, ts), ...]
    关键点：列表按时间倒序排列（最近发生的在前），方便快速检索最近历史。
    """
    graph: Dict[int, List[History]] = defaultdict(list)
    for head, relation, tail, ts in triples:
        graph[head].append((tail, relation, ts))

    # 对每个实体的历史按时间降序排序
    for history in graph.values():
        history.sort(key=lambda item: item[2], reverse=True)
    return graph


class TemporalDataset(Dataset):
    """
    PyTorch 数据集类。
    负责将原始三元组转换为模型所需的输入 Tensor（包含历史上下文）。
    """

    def __init__(
            self,
            triples: List[Triple],
            graph_index: Dict[int, List[History]],
            num_entities: int,
            num_relations: int,
            max_history: int = 4,  # 每个样本最多包含几条历史
            min_history: int = 1,  # 历史少于此数的样本会被丢弃
    ) -> None:
        self.graph_index = graph_index
        self.max_history = max_history
        self.min_history = min_history
        self.pad_entity = num_entities  # 用于填充空白历史的占位符 ID
        self.pad_relation = num_relations  # 用于填充空白关系的占位符 ID

        self.samples: List[Tuple[int, List[int], List[int], List[float], int, int, float]] = []

        # 预处理所有样本：为每个事件查找其对应的历史
        for head, relation, tail, ts in triples:
            history_entities, history_relations, history_deltas = self._collect_history(
                head, relation, ts
            )
            # 检查有效历史数量是否达标
            valid = sum(ent != self.pad_entity for ent in history_entities)
            if valid < self.min_history:
                continue

            # 保存处理好的样本元组
            self.samples.append(
                (
                    head,
                    history_entities,
                    history_relations,
                    history_deltas,
                    relation,
                    tail,
                    ts,
                )
            )

        if not self.samples:
            raise RuntimeError(
                "No samples have enough history; try reducing --min_history or increasing --max_history."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        获取第 idx 个样本，并将所有数据转换为 PyTorch Tensor。
        """
        head, history_entities, history_relations, history_deltas, relation, tail, ts = self.samples[idx]
        return {
            "head": torch.tensor(head, dtype=torch.long),
            "history_entities": torch.tensor(history_entities, dtype=torch.long),
            "history_relations": torch.tensor(history_relations, dtype=torch.long),
            "history_deltas": torch.tensor(history_deltas, dtype=torch.float),  # 时间差需为浮点数
            "query_relation": torch.tensor(relation, dtype=torch.long),
            "positive_tail": torch.tensor(tail, dtype=torch.long),  # 正样本（真实尾实体）
            "timestamp": torch.tensor(ts, dtype=torch.float),
        }

    def _collect_history(
            self, head: int, relation: int, current_ts: float
    ) -> Tuple[List[int], List[int], List[float]]:
        """
        核心逻辑：回溯历史。
        给定当前时刻 current_ts，查找该时刻之前发生的所有事件。
        """
        # 初始化：当前查询本身作为历史的第一项（Context）
        entities = [head]
        relations = [relation]
        deltas = [0.0]

        # 查档案
        for tail, rel, ts in self.graph_index.get(head, []):
            # 【关键】严禁穿越：只取当前时刻之前的事件
            if ts >= current_ts:
                continue
            entities.append(tail)
            relations.append(rel)
            deltas.append(max(0.0, current_ts - ts))  # 计算时间差

            # 截断：够了就停
            if len(entities) >= self.max_history:
                break

        # 填充 (Padding)：不够就补
        pad_len = self.max_history - len(entities)
        if pad_len > 0:
            entities.extend([self.pad_entity] * pad_len)
            relations.extend([self.pad_relation] * pad_len)
            deltas.extend([0.0] * pad_len)

        return entities, relations, deltas


def negative_sample(tails: torch.Tensor, num_entities: int) -> torch.Tensor:
    """
    负采样函数。
    随机生成一批错误的尾实体 ID，用于训练模型区分正负样本。
    """
    device = tails.device
    # 随机生成 ID
    neg = torch.randint(0, num_entities, size=tails.shape, dtype=torch.long, device=device)

    # 检查生成的负样本是否意外撞上了正样本（Mask），如果是，重新生成那一小部分
    mask = neg.eq(tails)
    while mask.any():
        neg[mask] = torch.randint(0, num_entities, size=(mask.sum().item(),), dtype=torch.long, device=device)
        mask = neg.eq(tails)
    return neg


def build_dataloader(args: argparse.Namespace) -> Tuple[List[Triple], TemporalDataset]:
    """
    数据流水线总入口：加载文件 -> 构建索引 -> 实例化数据集。
    """
    edge_file = Path(args.edge_file)
    # 推断数据集目录
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else infer_dataset_dir(edge_file)

    # 加载 ID 映射字典
    entity_map = load_mapping(dataset_dir / "entity2id.json")
    relation_map = load_mapping(dataset_dir / "relation2id.json")

    # 加载三元组数据
    triples = load_temporal_triples(dataset_dir, entity_map, relation_map, args.sample_limit or None)

    # 如果没读到，尝试备用 CSV 加载
    if not triples:
        print("[Warning] falling back to edge CSV because temporal splits were not found.")
        triples = fallback_load_from_csv(edge_file, args.sample_limit or None)
    if not triples:
        raise RuntimeError("No triples available for training.")

    # 构建图索引
    graph_index = build_graph_index(triples)

    # 创建 Dataset
    dataset = TemporalDataset(
        triples=triples,
        graph_index=graph_index,
        num_entities=args.num_entities,
        num_relations=args.num_relations,
        max_history=args.max_history,
        min_history=args.min_history,
    )
    print(
        f"[TemporalDataset] kept {len(dataset)} samples (history >= {args.min_history}) "
        f"from {len(triples)} events."
    )
    return triples, dataset


def train(args: argparse.Namespace) -> None:
    """
    训练主循环。
    """
    # 1. 准备数据
    _, dataset = build_dataloader(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # 2. 准备设备 (GPU/CPU)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 3. 初始化模型
    model = TemporalSARL(
        num_entities=args.num_entities + 1,  # +1 用于 Padding
        num_relations=args.num_relations + 1,
        embed_dim=args.embed_dim,
    ).to(device)

    # 4. 定义优化器 (Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 5. 开始 Epoch 循环
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            # 将数据搬运到 GPU
            head = batch["head"].to(device)
            history_entities = batch["history_entities"].to(device)
            history_relations = batch["history_relations"].to(device)
            history_deltas = batch["history_deltas"].to(device)
            query_relation = batch["query_relation"].to(device)
            positive_tail = batch["positive_tail"].to(device)

            zeros_delta = torch.zeros((head.size(0), 1), dtype=torch.float32, device=device)

            # --- 正样本前向传播 ---
            pos_scores = model(
                history_entities,
                history_relations,
                history_deltas,
                head,
                query_relation,
                positive_tail.unsqueeze(1),  # 正确答案
                query_relation.unsqueeze(1),
                zeros_delta,
            )
            # 计算正样本损失（目标是接近 1）
            pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))

            # --- 负样本前向传播 ---
            neg_tail = negative_sample(positive_tail, args.num_entities).to(device)
            neg_scores = model(
                history_entities,
                history_relations,
                history_deltas,
                head,
                query_relation,
                neg_tail.unsqueeze(1),  # 错误答案
                query_relation.unsqueeze(1),
                zeros_delta,
            )
            # 计算负样本损失（目标是接近 0）
            neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))

            # --- 反向传播与更新 ---
            loss = pos_loss + neg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        print(f"[Epoch {epoch}] loss={avg_loss:.4f}")

    # 6. 保存模型权重
    torch.save(model.state_dict(), args.save_path)
    print(f"[Checkpoint] saved model to {args.save_path}")


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="Train the Temporal-SARL model.")
    parser.add_argument("--edge_file", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, default="")
    parser.add_argument("--num_entities", type=int, required=True)
    parser.add_argument("--num_relations", type=int, required=True)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_history", type=int, default=5)
    parser.add_argument("--min_history", type=int, default=1)
    parser.add_argument("--sample_limit", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_path", type=str, default="./sarl_model.pth")
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())