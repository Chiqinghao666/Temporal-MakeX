#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 Makex SARL 全局模式的中文可读报告。"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, List, Sequence


def default_paths() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    makex_root = repo_root / "Makex-main"
    return argparse.Namespace(
        rep_file=makex_root / "global_explanations/rep_sarl.txt",
        relation_file=makex_root / "DataSets/icews14/relation2id.json",
        output_file=makex_root / "global_explanations/global_rules_readable.txt",
    )


def load_relation_map(path: Path) -> Dict[int, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {int(idx): name.replace("_", " ") for name, idx in data.items()}


def describe_relation(rel: int | str, rel_map: Dict[int, str]) -> str:
    """
    兼容字符串关系名与数值 ID，优先输出可读名称。
    """
    if isinstance(rel, str) and not rel.isdigit():
        return f"<{rel}>"
    try:
        rel_id = int(rel)
    except (TypeError, ValueError):
        return f"<{rel}>"
    name = rel_map.get(rel_id, f"关系{rel_id}")
    return f"<{rel_id}: {name}>"


def parse_rep(path: Path) -> List[dict]:
    patterns: List[dict] = []
    if not path.exists():
        return patterns
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = ast.literal_eval(line)
            except Exception:
                continue
            # 新格式（dict）直接保留
            if isinstance(entry, dict) and "nodes" in entry and "edges" in entry:
                patterns.append(entry)
                continue
            # 兼容旧格式
            if isinstance(entry, (list, tuple)) and len(entry) >= 4:
                patterns.append(
                    {
                        "nodes": entry[0],
                        "edges": entry[1],
                        "predicates": entry[2] if len(entry) > 2 else [],
                        "stats": entry[3],
                    }
                )
    return patterns


def split_stars(nodes: Sequence[Sequence[int]], edges: Sequence[Sequence[int]]):
    node_map = {int(node[0]): int(node[1]) for node in nodes}
    node_ids = list(node_map.keys())
    pivot_x = node_ids[0] if node_ids else 1
    pivot_y = node_ids[1] if len(node_ids) > 1 else pivot_x
    user_edges, item_edges = [], []
    for edge in edges:
        if len(edge) < 3:
            continue
        src, dst, _ = edge[:3]
        if src == pivot_x:
            user_edges.append(edge)
        elif src == pivot_y or dst == pivot_y:
            item_edges.append(edge)
    return node_map, pivot_x, pivot_y, user_edges, item_edges


def summarize_edges(edges: Sequence[Sequence[int]], role: str, rel_map: Dict[int, str]) -> List[str]:
    lines: List[str] = []
    for edge in edges:
        if len(edge) < 3:
            continue
        src, dst, rel = edge[:3]
        lines.append(
            f"  ({role}节点{src}) --[{describe_relation(rel, rel_map)}]--> (节点{dst})"
        )
    if not lines:
        lines.append(f"  ({role}) 无明确路径约束")
    return lines


def extract_query_relation(predicates: Sequence[Sequence]) -> tuple[int | None, str | None]:
    """
    同时提取关系 ID 与关系名称，优先返回名称。
    """
    rel_id = None
    rel_name = None
    for predicate in predicates:
        if len(predicate) >= 4 and predicate[0] == "Constant":
            key = str(predicate[2]).lower()
            if key == "query_relation":
                try:
                    rel_id = int(predicate[3])
                except ValueError:
                    rel_name = str(predicate[3])
            if key == "query_relation_id":
                try:
                    rel_id = int(predicate[3])
                except ValueError:
                    pass
            if key == "relation_chain" and not rel_name:
                rel_name = str(predicate[3])
    return rel_id, rel_name


def semantic_summary(edges: Sequence[Sequence[int]], rel_map: Dict[int, str]) -> str:
    relations = [describe_relation(edge[2], rel_map) for edge in edges if len(edge) >= 3]
    if not relations:
        return "此模式描述了节点间的结构性约束。"
    chain = " → ".join(relations[:4])
    if len(relations) > 4:
        chain += " → ..."
    return f"此模式捕捉了 “{chain}” 的时序互动。"


def build_report(patterns: List[dict], rel_map: Dict[int, str]) -> str:
    lines: List[str] = []
    for idx, pattern in enumerate(patterns, 1):
        # 新格式（dict）
        if isinstance(pattern, dict) and "query" in pattern and "nodes" in pattern and isinstance(pattern.get("edges"), list):
            support = int(pattern.get("support", 0))
            conf = float(pattern.get("confidence", 0.0))
            query_info = pattern.get("query", {})
            lines.append(f"[模式 ID: {idx}]")
            lines.append(f"统计: 支持度 {support} | 置信度 {conf * 100:.1f}%")
            lines.append(
                f"意图: 预测关系 <{query_info.get('relation', query_info.get('relation_id', ''))}> | 星侧: {query_info.get('star_side', '')}"
            )

            node_desc = []
            for n in pattern["nodes"]:
                node_desc.append(
                    f"节点{n.get('pattern_id')}: {n.get('name')} (ID={n.get('entity_id')}, 类型={n.get('type')})"
                )
            lines.append("节点: " + "; ".join(node_desc))

            edge_desc = []
            for e in pattern["edges"]:
                edge_desc.append(
                    f"{e.get('src')} -[{e.get('relation')}]→ {e.get('dst')} (time_bin={e.get('time_bin')}, gap_days={e.get('gap_days')})"
                )
            lines.append("路径: " + " | ".join(edge_desc))
            lines.append("")
            continue

        # 兼容旧格式
        stats = pattern["stats"] if pattern.get("stats") else [0, 0]
        support = int(stats[0]) if stats else 0
        conf = float(stats[1]) if len(stats) > 1 else 0.0
        node_map, pivot_x, pivot_y, user_edges, item_edges = split_stars(
            pattern["nodes"], pattern["edges"]
        )
        query_rel_id, query_rel_name = extract_query_relation(pattern.get("predicates", []))
        if query_rel_name:
            intent = f"专门用于预测关系 <{query_rel_name}>"
        elif query_rel_id is not None:
            intent = f"专门用于预测关系 {describe_relation(query_rel_id, rel_map)}"
        else:
            intent = "用于捕捉目标节点之间的共现结构"
        lines.append(f"[模式 ID: {idx}]")
        lines.append(f"统计: 支持度 {support} | 置信度 {conf * 100:.1f}%")
        lines.append(f"意图: {intent}")
        lines.append("结构:")
        lines.extend(summarize_edges(user_edges, "User", rel_map))
        lines.extend(summarize_edges(item_edges, "Item", rel_map))
        lines.append("语义:")
        lines.append(f"  {semantic_summary(pattern['edges'], rel_map)}")
        lines.append("")
    if not lines:
        lines.append("未找到任何模式。")
    return "\n".join(lines)


def main() -> None:
    defaults = default_paths()
    parser = argparse.ArgumentParser(description="生成全局时态模式报告")
    parser.add_argument("--rep_file", type=Path, default=defaults.rep_file)
    parser.add_argument("--relation_file", type=Path, default=defaults.relation_file)
    parser.add_argument("--output_file", type=Path, default=defaults.output_file)
    args = parser.parse_args()

    patterns = parse_rep(args.rep_file)
    rel_map = load_relation_map(args.relation_file)
    report = build_report(patterns, rel_map)

    output_path = args.output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"《全局时态模式报告》已生成至 {output_path}")


if __name__ == "__main__":
    main()
