#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main entry for Temporal-SARL discovery (mining)."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

import pyMakex  # type: ignore
from sarl_model import TemporalSARL
from temporal_sarl_miner import SARLMiner, SARLOptions, TemporalNeighbor, TemporalPath

Triple = Tuple[int, int, int, float]


def load_json(path: Path) -> Dict[str, int]:
    return json.loads(path.read_text())


def parse_temporal_line(
    line: str,
    entity_map: Dict[str, int],
    relation_map: Dict[str, int],
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
    return entity_map[head], relation_map[relation], entity_map[tail], ts


def load_temporal_triples(
    dataset_dir: Path,
    entity_map: Dict[str, int],
    relation_map: Dict[str, int],
    limit: int | None,
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


def build_edge_store(triples: Sequence[Triple]) -> Dict[int, List[TemporalNeighbor]]:
    store: Dict[int, List[TemporalNeighbor]] = {}
    for head, relation, tail, ts in triples:
        store.setdefault(head, []).append(TemporalNeighbor(dst=tail, relation=relation, timestamp=ts))
    for edges in store.values():
        edges.sort(key=lambda e: e.timestamp, reverse=True)
    return store


def sample_queries(triples: Sequence[Triple], num_queries: int) -> List[Triple]:
    triples = list(triples)
    random.shuffle(triples)
    if num_queries >= len(triples):
        return triples
    return triples[:num_queries]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute Temporal-SARL mining on ICEWS.")
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
    parser.add_argument("--entity_map", type=Path, default=Path("../DataSets/icews14/entity2id.json"))
    parser.add_argument("--relation_map", type=Path, default=Path("../DataSets/icews14/relation2id.json"))
    parser.add_argument("--model_path", type=Path, default=Path("./sarl_model.pth"))
    parser.add_argument("--num_queries", type=int, default=200)
    parser.add_argument("--walks_per_query", type=int, default=50)
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--beam_size", type=int, default=8)
    parser.add_argument("--history_size", type=int, default=5)
    parser.add_argument("--time_window", type=float, default=30 * 86400.0)
    parser.add_argument("--time_bucket", type=float, default=7 * 86400.0)
    parser.add_argument("--top_signatures", type=int, default=200)
    parser.add_argument("--sample_limit", type=int, default=0)
    parser.add_argument("--output_rep", type=Path, default=Path("./rep_sarl.txt"))
    parser.add_argument("--log_dir", type=Path, default=Path("./global_explanations"))
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=1996)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    entity_map = load_json(args.entity_map)
    relation_map = load_json(args.relation_map)
    triples = load_temporal_triples(args.dataset_dir, entity_map, relation_map, args.sample_limit or None)
    if not triples:
        raise RuntimeError("No temporal triples found; ensure ICEWS train/valid/test exist.")

    edge_store = build_edge_store(triples)
    queries = sample_queries(triples, args.num_queries)
    print(f"[Init] Loaded {len(triples)} triples, sampled {len(queries)} queries.")

    num_entities = len(entity_map) + 1
    num_relations = len(relation_map) + 1
    model = TemporalSARL(num_entities=num_entities, num_relations=num_relations)
    state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict)

    graph_ptr = pyMakex.ReadDataGraph(str(args.vertex_file), str(args.edge_file))
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    options = SARLOptions(
        max_hops=args.max_hops,
        history_size=args.history_size,
        beam_size=args.beam_size,
        time_window=args.time_window,
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
    miner.reset_statistics()

    all_paths: List[TemporalPath] = []
    total_queries = len(queries)
    for idx, (head, relation, _, ts) in enumerate(queries, 1):
        if idx % 10 == 0 or idx == total_queries:
            print(f"[Progress] Processed {idx}/{total_queries} queries...")
        paths = miner.mine_paths(head, relation, ts, num_walks=args.walks_per_query)
        all_paths.extend(paths)

    miner.report_performance()
    if not all_paths:
        print("[Warning] No paths mined; rep file will not be created.")
        return

    grouped = miner.cluster_paths(all_paths, args.time_bucket)
    ranked = sorted(grouped.items(), key=lambda kv: len(kv[1]), reverse=True)
    rep_entries = []
    for (head, relation, _), path_group in ranked[: args.top_signatures]:
        rep_entries.append(miner.path_to_rep(path_group[0], len(path_group)))

    with args.output_rep.open("w", encoding="utf-8") as fp:
        for entry in rep_entries:
            fp.write(f"{entry}\n")
    print(
        f"[Summary] Saved {len(rep_entries)} SARL patterns to {args.output_rep}."
        f" Raw paths logged at {options.log_dir / 'sarl_raw_paths.txt'}"
    )


if __name__ == "__main__":
    main()
