#!/usr/bin/env python3
"""将 Makex 生成的全局 REP 规则翻译为中文可读文字。"""

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def default_paths() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    makex_root = repo_root / "Makex-main"
    return argparse.Namespace(
        rep_file=makex_root / "global_explanations/rep.txt",
        edge_file=makex_root
        / "DataSets/icews14/processed/original_graph/icews_e.csv",
        vertex_file=makex_root
        / "DataSets/icews14/processed/original_graph/icews_v.csv",
        relation_file=makex_root / "DataSets/icews14/relation2id.json",
    )


def load_relation_map(path: Path) -> Dict[int, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {int(idx): name.replace("_", " ") for name, idx in data.items()}


def load_label_examples(vertex_path: Path) -> Dict[int, str]:
    examples: Dict[int, str] = {}
    if not vertex_path.exists():
        return examples
    with vertex_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_key = row.get("label_id:int") or row.get("label_id")
            name = row.get("name:string") or row.get("name") or ""
            if label_key is None:
                continue
            label = int(label_key)
            if label not in examples and name:
                examples[label] = name
    return examples


def describe_label(label_id: int, examples: Dict[int, str]) -> str:
    if label_id in examples:
        return f"标签{label_id}（示例：{examples[label_id]}）"
    return f"标签{label_id}"


def describe_relation(rel_id: int, relation_map: Dict[int, str]) -> str:
    name = relation_map.get(rel_id)
    if not name:
        return f"关系{rel_id}"
    return f"{name}（ID: {rel_id}）"


def describe_predicates(predicates: Sequence[Sequence]) -> List[str]:
    readable: List[str] = []
    for predicate in predicates:
        if not predicate:
            continue
        kind = predicate[0]
        if kind == "Constant" and len(predicate) >= 6:
            _, node_id, attr, value, _, op = predicate[:6]
            readable.append(f"Pattern 节点{node_id} 的 {attr} {op} {value}")
        elif kind == "Variable" and len(predicate) >= 6:
            _, x_node, x_attr, y_node, y_attr, op = predicate[:6]
            readable.append(
                f"Pattern 节点{x_node} 的 {x_attr} {op} 节点{y_node} 的 {y_attr}"
            )
        else:
            readable.append(f"谓词 {predicate}")
    if not readable:
        readable.append("（无）")
    return readable


def split_stars(
    edges: Sequence[Sequence[int]],
    node_info: Dict[int, int],
) -> Tuple[List[Sequence[int]], List[Sequence[int]], int, int]:
    node_ids = list(node_info.keys())
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
    return user_edges, item_edges, pivot_x, pivot_y


def describe_edges(
    star_edges: Sequence[Sequence[int]],
    node_info: Dict[int, int],
    relation_map: Dict[int, str],
    label_examples: Dict[int, str],
) -> List[str]:
    lines: List[str] = []
    for idx, edge in enumerate(star_edges, 1):
        if len(edge) < 3:
            lines.append(f"- 路径 {idx}: {edge}")
            continue
        src, dst, rel = edge[:3]
        src_label = describe_label(node_info.get(src, -1), label_examples)
        dst_label = describe_label(node_info.get(dst, -1), label_examples)
        relation_name = describe_relation(rel, relation_map)
        lines.append(
            f"- 路径 {idx}: Pattern 节点{src}（{src_label}） --[{relation_name}]--> "
            f"节点{dst}（{dst_label}）"
        )
    if not lines:
        lines.append("（暂无路径）")
    return lines


def main() -> None:
    defaults = default_paths()
    parser = argparse.ArgumentParser(
        description="读取 rep.txt 并输出中文解释"
    )
    parser.add_argument(
        "--rep_file", type=Path, default=defaults.rep_file, help="rep.txt 路径"
    )
    parser.add_argument(
        "--vertex_file",
        type=Path,
        default=defaults.vertex_file,
        help="顶点 CSV，提供标签示例",
    )
    parser.add_argument(
        "--relation_file",
        type=Path,
        default=defaults.relation_file,
        help="relation2id.json，用于翻译交互类型",
    )
    args = parser.parse_args()

    if not args.rep_file.exists():
        raise SystemExit(f"未找到 rep 文件：{args.rep_file}")

    relation_map = load_relation_map(args.relation_file)
    label_examples = load_label_examples(args.vertex_file)

    print("=== Makex 全局规则解释 ===")
    with args.rep_file.open() as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rep_entry = ast.literal_eval(line)
            except Exception as err:
                print(f"[警告] 第 {idx} 行解析失败：{err}")
                continue
            if len(rep_entry) < 4:
                continue
            nodes = rep_entry[0]
            edges = rep_entry[1]
            predicates = rep_entry[2] if len(rep_entry) > 2 else []
            stats = rep_entry[3]
            support = stats[0] if stats else 0.0
            conf = stats[1] if len(stats) > 1 else 0.0
            node_info = {int(node[0]): int(node[1]) for node in nodes}
            user_edges, item_edges, pivot_x, pivot_y = split_stars(
                edges, node_info
            )
            predicate_text = describe_predicates(predicates)
            print(
                f"\n=== 全局规则 Pattern #{idx} (支持度: {int(support)}, "
                f"置信度: {conf * 100:.1f}%) ==="
            )
            print("【双星结构 (Dual-Star Structure)】")
            print(f"  ★ 用户星 (以 Pattern 节点{pivot_x} 为中心):")
            for text in describe_edges(
                user_edges, node_info, relation_map, label_examples
            ):
                print(f"     {text}")
            print(f"  ★ 物品星 (以 Pattern 节点{pivot_y} 为中心):")
            for text in describe_edges(
                item_edges, node_info, relation_map, label_examples
            ):
                print(f"     {text}")
            print("\n【属性限制 (Predicates)】")
            for text in predicate_text:
                print(f"  - {text}")
            print("\n【解释】")
            print(
                "  当用户星与物品星同时匹配上述结构，并满足属性限制时，本规则触发。"
            )
            print("-" * 48)


if __name__ == "__main__":
    main()
