#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 Makex SARL 局部预测与归因报告。"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def default_paths() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    makex_root = repo_root / "Makex-main"
    return argparse.Namespace(
        local_file=makex_root / "local_explanations/output/icews/local_topk1.txt",
        rep_file=makex_root / "global_explanations/rep_sarl.txt",
        vertex_file=makex_root
        / "DataSets/icews14/processed/original_graph/icews_v.csv",
        relation_file=makex_root / "DataSets/icews14/relation2id.json",
        topk_file=makex_root / "local_explanations/output/icews/topk_rep_id_topk1.csv",
        pairs_file=makex_root / "local_explanations/output/icews/test_sample_pairs.csv",
        subgraph_file=makex_root / "local_explanations/output/icews/subgraph.csv",
        output_file=makex_root
        / "local_explanations/output/icews/prediction_report.txt",
    )


def parse_log_params(log_path: Path) -> Dict[str, str]:
    params: Dict[str, str] = {}
    if not log_path.exists():
        return params
    with log_path.open() as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            params[key.strip()] = value.strip()
    return params


def resolve_relative(raw: str, base_dir: Path, root_dir: Path) -> Path:
    if not raw:
        return Path()
    path = Path(raw)
    if path.is_absolute():
        return path
    if raw.startswith("./"):
        return (root_dir / raw[2:]).resolve()
    return (base_dir / raw).resolve()


def load_vertex_names(vertex_path: Path) -> Dict[int, str]:
    id_to_name: Dict[int, str] = {}
    if not vertex_path.exists():
        return id_to_name
    with vertex_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid_key = row.get("vertex_id:int") or row.get("vertex_id")
            name = row.get("name:string") or row.get("name") or ""
            if vid_key is None:
                continue
            vid = int(vid_key)
            id_to_name[vid] = name or f"实体{vid}"
    return id_to_name


def load_relation_map(path: Path) -> Dict[int, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {int(idx): name.replace("_", " ") for name, idx in data.items()}


def load_pairs(pairs_path: Path) -> Dict[int, Tuple[int, int]]:
    pairs: Dict[int, Tuple[int, int]] = {}
    if not pairs_path.exists():
        return pairs
    with pairs_path.open() as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            user = int(row["user_id"])
            item = int(row["item_id"])
            pairs[idx] = (user, item)
    return pairs


def load_query_relations(subgraph_path: Path) -> Dict[int, int]:
    relation_by_pair: Dict[int, int] = {}
    if not subgraph_path.exists():
        return relation_by_pair
    with subgraph_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            gid = int(row["graph_id"])
            relation_by_pair[gid] = int(row["edge_type"])
    return relation_by_pair


def load_topk(topk_path: Path) -> List[dict]:
    entries: List[dict] = []
    if not topk_path.exists():
        return entries
    with topk_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {
                "pair_id": int(row["pair_id"]),
                "pivot_x": int(row["pivot_x"]),
                "pivot_y": int(row["pivot_y"]),
                "rank": int(row.get("topk", row.get("rank", 0))) + 1,
                "rep_id": int(row.get("rep_id", 0)),
                "score": row.get("score", "N/A"),
            }
            entries.append(entry)
    return entries


def load_patterns(rep_path: Path) -> List[dict]:
    patterns: List[dict] = []
    if not rep_path.exists():
        return patterns
    with rep_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = ast.literal_eval(line)
            except Exception:
                continue
            if len(entry) < 4:
                continue
            patterns.append(
                {
                    "nodes": entry[0],
                    "edges": entry[1],
                    "predicates": entry[2] if len(entry) > 2 else [],
                    "stats": entry[3],
                }
            )
    return patterns


def describe_relation(rel_id: int, rel_map: Dict[int, str]) -> str:
    name = rel_map.get(rel_id, f"关系{rel_id}")
    return f"<{rel_id}: {name}>"


def split_stars(nodes, edges):
    node_map = {int(node[0]): int(node[1]) for node in nodes}
    node_ids = list(node_map.keys())
    pivot_x = node_ids[0] if node_ids else 1
    pivot_y = node_ids[1] if len(node_ids) > 1 else pivot_x
    user_edges, item_edges = [], []
    for edge in edges:
        if len(edge) < 3:
            continue
        src, dst, rel = edge[:3]
        if src == pivot_x:
            user_edges.append(edge)
        elif src == pivot_y or dst == pivot_y:
            item_edges.append(edge)
    return user_edges, item_edges


def summarize_pattern(pattern: dict, rel_map: Dict[int, str]) -> Tuple[str, str]:
    edges = pattern.get("edges", [])
    relations = [describe_relation(edge[2], rel_map) for edge in edges if len(edge) >= 3]
    if not relations:
        return ("结构特征：该模式仅包含节点属性约束。", "语义：描述目标实体间的共现。")
    chain = " → ".join(relations[:4])
    if len(relations) > 4:
        chain += " → ..."
    return (
        f"结构特征：{chain}",
        "语义：两端实体通过上述互动形成时序闭环。",
    )


def explain_prediction(
    pattern: dict,
    rel_map: Dict[int, str],
    user_name: str,
    item_name: str,
) -> List[str]:
    user_edges, item_edges = split_stars(pattern["nodes"], pattern["edges"])
    user_rel = [describe_relation(edge[2], rel_map) for edge in user_edges if len(edge) >= 3]
    item_rel = [describe_relation(edge[2], rel_map) for edge in item_edges if len(edge) >= 3]
    lines: List[str] = []
    if user_rel:
        lines.append(
            f"  因为 {user_name} 在用户星中近期涉及 {"、".join(user_rel)} 等互动"
        )
    if item_rel:
        lines.append(
            f"  同时 {item_name} 在物品星中关联 {"、".join(item_rel)} 等行为"
        )
    if not lines:
        lines.append(
            "  该模式主要通过节点属性与结构约束解释本次预测"
        )
    lines.append("  两者共同满足该模式的多跳结构，从而获得较高置信度。")
    return lines


def build_report(
    entries: List[dict],
    pairs: Dict[int, Tuple[int, int]],
    relation_map: Dict[int, str],
    query_relations: Dict[int, int],
    vertex_names: Dict[int, str],
    patterns: List[dict],
) -> str:
    lines: List[str] = []
    entries.sort(key=lambda e: (e["pair_id"], e["rank"]))
    grouped: Dict[int, List[dict]] = {}
    for entry in entries:
        grouped.setdefault(entry["pair_id"], []).append(entry)

    for pair_id in sorted(grouped.keys()):
        user_id, item_id = pairs.get(pair_id, (None, None))
        if user_id is None or item_id is None:
            continue
        user_name = vertex_names.get(user_id, f"实体{user_id}")
        item_name = vertex_names.get(item_id, f"实体{item_id}")
        relation_id = query_relations.get(pair_id)
        relation_desc = describe_relation(relation_id, relation_map) if relation_id is not None else "<未知关系>"
        lines.append("=" * 60)
        lines.append(
            f"查询事件: (User: {user_name}, Relation: {relation_desc}, Target: ?, Time: N/A)"
        )
        lines.append("=" * 60)
        lines.append("")
        for entry in grouped[pair_id]:
            rep_id = entry["rep_id"]
            pattern_idx = rep_id + 1
            pattern = patterns[rep_id] if 0 <= rep_id < len(patterns) else None
            score = entry["score"] if entry["score"] not in (None, "", "N/A") else "N/A"
            pred_name = vertex_names.get(entry["pivot_y"], f"实体{entry['pivot_y']}")
            lines.append(
                f"> 排名 {entry['rank']} 预测: {pred_name} (得分: {score})"
            )
            if pattern is None:
                lines.append("  [解释依据] 模式信息缺失。")
            else:
                struct_desc, semantic_desc = summarize_pattern(pattern, relation_map)
                lines.append(f"  [解释依据 - 匹配模式 #{pattern_idx}]")
                lines.append(f"  {struct_desc}")
                lines.append(f"  {semantic_desc}")
                lines.extend(
                    explain_prediction(
                        pattern,
                        relation_map,
                        user_name,
                        pred_name,
                    )
                )
            lines.append("")
    if not lines:
        lines.append("未找到任何预测记录。")
    return "\n".join(lines)


def main() -> None:
    defaults = default_paths()
    parser = argparse.ArgumentParser(description="生成时态预测与归因报告")
    parser.add_argument("--local_file", type=Path, default=defaults.local_file)
    parser.add_argument("--rep_file", type=Path, default=defaults.rep_file)
    parser.add_argument("--vertex_file", type=Path, default=defaults.vertex_file)
    parser.add_argument("--relation_file", type=Path, default=defaults.relation_file)
    parser.add_argument("--topk_file", type=Path, default=defaults.topk_file)
    parser.add_argument("--pairs_file", type=Path, default=defaults.pairs_file)
    parser.add_argument("--subgraph_file", type=Path, default=defaults.subgraph_file)
    parser.add_argument("--output_file", type=Path, default=defaults.output_file)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    makex_root = repo_root / "Makex-main"
    local_root = makex_root / "local_explanations"

    log_params = parse_log_params(args.local_file)
    base_dir = args.local_file.parent if args.local_file else Path(".")
    if log_params.get("topk_rep_id_file"):
        args.topk_file = resolve_relative(log_params["topk_rep_id_file"], base_dir, local_root)
    if log_params.get("test_pairs_file"):
        args.pairs_file = resolve_relative(log_params["test_pairs_file"], base_dir, local_root)
    if log_params.get("subgraph_path"):
        args.subgraph_file = resolve_relative(log_params["subgraph_path"], base_dir, local_root)

    vertex_names = load_vertex_names(args.vertex_file)
    relation_map = load_relation_map(args.relation_file)
    pairs = load_pairs(args.pairs_file)
    query_relations = load_query_relations(args.subgraph_file)
    topk_entries = load_topk(args.topk_file)
    patterns = load_patterns(args.rep_file)

    if not topk_entries:
        raise SystemExit(f"未找到 topk 预测文件：{args.topk_file}")

    report = build_report(
        topk_entries,
        pairs,
        relation_map,
        query_relations,
        vertex_names,
        patterns,
    )

    output_path = args.output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"《时态预测与归因报告》已生成至 {output_path}")


if __name__ == "__main__":
    main()
