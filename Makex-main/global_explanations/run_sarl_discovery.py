#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Temporal SARL 全流程挖掘脚本。"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

import pyMakex  # type: ignore
from sarl_model import TemporalSARL
from temporal_sarl_miner import SARLMiner, SARLOptions, TemporalNeighbor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Temporal SARL discovery.")
    parser.add_argument("--dataset_dir", type=Path, default=Path("../DataSets/icews14"))
    parser.add_argument(
        "--vertex_file",
        type=Path,
        default=Path("../DataSets/icews14/processed/original_graph/icews_v.csv"),
    )
    parser.add_argument(
        "--edge_file",
        type=Path,
        default=Path("../DataSets/icews14/processed/original_graph/icews_e.csv"),
    )
    parser.add_argument("--model_path", type=Path, default=Path("./sarl_model.pth"))
    parser.add_argument(
        "--entity_map", type=Path, default=Path("../DataSets/icews14/entity2id.json")
    )
    parser.add_argument(
        "--relation_map", type=Path, default=Path("../DataSets/icews14/relation2id.json")
    )
    parser.add_argument("--output_rep", type=Path, default=Path("./rep_sarl.txt"))
    parser.add_argument("--num_entities", type=int, default=0)
    parser.add_argument("--num_relations", type=int, default=0)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_queries", type=int, default=200)
    parser.add_argument("--walks_per_query", type=int, default=50)
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--beam_size", type=int, default=8)
    parser.add_argument("--time_window", type=float, default=86400.0)
    parser.add_argument("--top_signatures", type=int, default=200)
    parser.add_argument("--time_bucket", type=float, default=86400.0)
    parser.add_argument("--sample_limit", type=int, default=0)
    parser.add_argument("--log_dir", type=Path, default=CURRENT_DIR)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, int]:
    return json.loads(path.read_text())


def parse_temporal_line(
    line: str, entity_map: Dict[str, int], relation_map: Dict[str, int]
) -> Tuple[int, int, int, float] | None:
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
    limit: int | None = None,
) -> List[Tuple[int, int, int, float]]:
    triples: List[Tuple[int, int, int, float]] = []
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


def build_edge_store(
    triples: Sequence[Tuple[int, int, int, float]]
) -> Dict[int, List[TemporalNeighbor]]:
    store: Dict[int, List[TemporalNeighbor]] = defaultdict(list)
    for head, relation, tail, ts in triples:
        store[head].append(TemporalNeighbor(dst=tail, relation=relation, timestamp=ts))
    for edges in store.values():
        edges.sort(key=lambda e: e.timestamp, reverse=True)
    return store


def sample_queries(
    triples: Sequence[Tuple[int, int, int, float]], num_queries: int
) -> List[Tuple[int, int, int, float]]:
    triples_list = list(triples)
    random.shuffle(triples_list)
    return triples_list[: num_queries]


def build_signature(
    path: List[TemporalNeighbor], query_time: float, time_bucket: float
) -> str:
    tokens = [f"L{len(path)}"]
    for edge in path:
        delta = max(0.0, query_time - edge.timestamp)
        bucket = int(delta // time_bucket)
        tokens.append(f"{edge.relation}:{bucket}")
    return "|".join(tokens)


def path_to_rep(
    head: int, relation: int, path: List[TemporalNeighbor], support: int
) -> List:
    node_index: Dict[int, int] = {head: 1}
    ordered_nodes = [head]
    current = head
    for edge in path:
        if edge.dst not in node_index:
            node_index[edge.dst] = len(node_index) + 1
            ordered_nodes.append(edge.dst)
        current = edge.dst
    vertex_list = [[idx + 1, 0] for idx in range(len(node_index))]
    edges = []
    current = head
    for edge in path:
        src_idx = node_index[current]
        dst_idx = node_index[edge.dst]
        edges.append([src_idx, dst_idx, edge.relation])
        current = edge.dst
    predicates = [
        ["Constant", 1, "query_relation", str(relation), "string", "="],
        ["Constant", 1, "path_length", str(len(path)), "string", "="],
    ]
    stats = [float(support), 1.0]
    meta = [1, 2, 1, 1.0]
    return [vertex_list, edges, predicates, stats, meta]


def main() -> None:
    args = parse_args()
    random.seed(1996)
    entity_map = load_json(args.entity_map)
    relation_map = load_json(args.relation_map)
    triples = load_temporal_triples(
        args.dataset_dir, entity_map, relation_map, args.sample_limit or None
    )
    if not triples:
        raise RuntimeError("未能从数据集中解析到任何三元组，无法执行 SARL。")
    edge_store = build_edge_store(triples)
    seed_queries = sample_queries(triples, args.num_queries)
    num_entities = args.num_entities or len(entity_map)
    num_relations = args.num_relations or len(relation_map)

    model = TemporalSARL(
        num_entities=num_entities,
        num_relations=num_relations,
        embed_dim=args.embed_dim,
    )
    state = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state)

    graph_ptr = pyMakex.ReadDataGraph(str(args.vertex_file), str(args.edge_file))
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    options = SARLOptions(
        max_hops=args.max_hops,
        beam_size=args.beam_size,
        time_decay=args.time_window,
        device=device,
        log_dir=args.log_dir,
    )
    miner = SARLMiner(
        model=model,
        options=options,
        entity2id_path=args.entity_map,
        relation2id_path=args.relation_map,
        graph_ptr=graph_ptr,
        edge_store=edge_store,
    )

    all_paths = []
    total_queries = len(seed_queries)
    for idx, (head, relation, _, ts) in enumerate(seed_queries, 1):
        if idx % 10 == 0 or idx == total_queries:
            print(f"[Progress] Processed {idx}/{total_queries} queries...")
        paths = miner.mine_paths(head, relation, ts, num_walks=args.walks_per_query)
        for path in paths:
            if path:
                all_paths.append({"head": head, "relation": relation, "query_time": ts, "path": path})

    if not all_paths:
        print("未挖掘到任何有效路径，rep 文件不会被更新。")
        return
    print(
        f"[Hit Rate] Queries={total_queries}, "
        f"Valid paths={len(all_paths)}, "
        f"Avg paths/query={len(all_paths) / max(total_queries,1):.2f}"
    )

    grouped: Dict[Tuple[int, int, str], List[List[TemporalNeighbor]]] = defaultdict(list)
    for entry in all_paths:
        signature = build_signature(entry["path"], entry["query_time"], args.time_bucket)
        key = (entry["head"], entry["relation"], signature)
        grouped[key].append(entry["path"])

    ranked_groups = sorted(grouped.items(), key=lambda kv: len(kv[1]), reverse=True)
    rep_entries = []
    for (head, relation, _), path_list in ranked_groups[: args.top_signatures]:
        rep_entries.append(path_to_rep(head, relation, path_list[0], len(path_list)))

    output_path = args.output_rep
    with output_path.open("w", encoding="utf-8") as f:
        for entry in rep_entries:
            f.write(f"{entry}\n")
    print(
        f"[Summary] Saved {len(rep_entries)} SARL patterns to {output_path}. "
        f"Raw paths logged to {options.log_dir / 'sarl_raw_paths.txt'}"
    )


if __name__ == "__main__":
    main()
