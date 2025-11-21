#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training script for Temporal-SARL.

Features:
- builds an in-memory temporal graph index from ICEWS splits
- samples real historical context for every query (monotonic timestamps)
- applies negative sampling per batch
- supports GPU training and checkpoint saving
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

Triple = Tuple[int, int, int, float]
History = Tuple[int, int, float]


def load_mapping(path: Path) -> Dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(f"Missing mapping file: {path}")
    return json.loads(path.read_text())


def infer_dataset_dir(edge_file: Path) -> Path:
    for parent in edge_file.resolve().parents:
        if (parent / "train.txt").exists() and (parent / "entity2id.json").exists():
            return parent
    raise RuntimeError("Unable to infer dataset directory; set --dataset_dir explicitly.")


def parse_temporal_line(
    line: str,
    entity_map: Dict[str, int],
    relation_map: Dict[str, int],
) -> Optional[Triple]:
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
) -> List[Triple]:
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
                if limit and len(triples) >= limit:
                    return triples
    return triples


def fallback_load_from_csv(edge_file: Path, limit: Optional[int]) -> List[Triple]:
    triples: List[Triple] = []
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


def build_graph_index(triples: List[Triple]) -> Dict[int, List[History]]:
    graph: Dict[int, List[History]] = defaultdict(list)
    for head, relation, tail, ts in triples:
        graph[head].append((tail, relation, ts))
    for history in graph.values():
        history.sort(key=lambda item: item[2], reverse=True)
    return graph


class TemporalDataset(Dataset):
    def __init__(
        self,
        triples: List[Triple],
        graph_index: Dict[int, List[History]],
        num_entities: int,
        num_relations: int,
        max_history: int = 4,
        min_history: int = 1,
    ) -> None:
        self.graph_index = graph_index
        self.max_history = max_history
        self.min_history = min_history
        self.pad_entity = num_entities
        self.pad_relation = num_relations
        self.samples: List[Tuple[int, List[int], List[int], List[float], int, int, float]] = []
        for head, relation, tail, ts in triples:
            history_entities, history_relations, history_deltas = self._collect_history(
                head, relation, ts
            )
            valid = sum(ent != self.pad_entity for ent in history_entities)
            if valid < self.min_history:
                continue
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
        head, history_entities, history_relations, history_deltas, relation, tail, ts = self.samples[idx]
        return {
            "head": torch.tensor(head, dtype=torch.long),
            "history_entities": torch.tensor(history_entities, dtype=torch.long),
            "history_relations": torch.tensor(history_relations, dtype=torch.long),
            "history_deltas": torch.tensor(history_deltas, dtype=torch.float),
            "query_relation": torch.tensor(relation, dtype=torch.long),
            "positive_tail": torch.tensor(tail, dtype=torch.long),
            "timestamp": torch.tensor(ts, dtype=torch.float),
        }

    def _collect_history(
        self, head: int, relation: int, current_ts: float
    ) -> Tuple[List[int], List[int], List[float]]:
        entities = [head]
        relations = [relation]
        deltas = [0.0]
        for tail, rel, ts in self.graph_index.get(head, []):
            if ts >= current_ts:
                continue
            entities.append(tail)
            relations.append(rel)
            deltas.append(max(0.0, current_ts - ts))
            if len(entities) >= self.max_history:
                break
        pad_len = self.max_history - len(entities)
        if pad_len > 0:
            entities.extend([self.pad_entity] * pad_len)
            relations.extend([self.pad_relation] * pad_len)
            deltas.extend([0.0] * pad_len)
        return entities, relations, deltas


def negative_sample(tails: torch.Tensor, num_entities: int) -> torch.Tensor:
    device = tails.device
    neg = torch.randint(0, num_entities, size=tails.shape, dtype=torch.long, device=device)
    mask = neg.eq(tails)
    while mask.any():
        neg[mask] = torch.randint(0, num_entities, size=(mask.sum().item(),), dtype=torch.long, device=device)
        mask = neg.eq(tails)
    return neg


def build_dataloader(args: argparse.Namespace) -> Tuple[List[Triple], TemporalDataset]:
    edge_file = Path(args.edge_file)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else infer_dataset_dir(edge_file)
    entity_map = load_mapping(dataset_dir / "entity2id.json")
    relation_map = load_mapping(dataset_dir / "relation2id.json")
    triples = load_temporal_triples(dataset_dir, entity_map, relation_map, args.sample_limit or None)
    if not triples:
        print("[Warning] falling back to edge CSV because temporal splits were not found.")
        triples = fallback_load_from_csv(edge_file, args.sample_limit or None)
    if not triples:
        raise RuntimeError("No triples available for training.")
    graph_index = build_graph_index(triples)
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
    _, dataset = build_dataloader(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = TemporalSARL(
        num_entities=args.num_entities + 1,
        num_relations=args.num_relations + 1,
        embed_dim=args.embed_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            head = batch["head"].to(device)
            history_entities = batch["history_entities"].to(device)
            history_relations = batch["history_relations"].to(device)
            history_deltas = batch["history_deltas"].to(device)
            query_relation = batch["query_relation"].to(device)
            positive_tail = batch["positive_tail"].to(device)

            zeros_delta = torch.zeros((head.size(0), 1), dtype=torch.float32, device=device)

            pos_scores = model(
                history_entities,
                history_relations,
                history_deltas,
                head,
                query_relation,
                positive_tail.unsqueeze(1),
                query_relation.unsqueeze(1),
                zeros_delta,
            )
            pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))

            neg_tail = negative_sample(positive_tail, args.num_entities).to(device)
            neg_scores = model(
                history_entities,
                history_relations,
                history_deltas,
                head,
                query_relation,
                neg_tail.unsqueeze(1),
                query_relation.unsqueeze(1),
                zeros_delta,
            )
            neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))

            loss = pos_loss + neg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        print(f"[Epoch {epoch}] loss={avg_loss:.4f}")

    torch.save(model.state_dict(), args.save_path)
    print(f"[Checkpoint] saved model to {args.save_path}")


def parse_args() -> argparse.Namespace:
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
