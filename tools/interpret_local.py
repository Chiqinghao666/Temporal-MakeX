#!/usr/bin/env python3
"""读取 Makex local_topk 输出并给出中文解释。"""

import argparse
import ast
import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def default_paths() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    makex_root = repo_root / "Makex-main"
    return argparse.Namespace(
        local_file=makex_root
        / "local_explanations/output/icews/local_topk1.txt",
        rep_file=makex_root / "global_explanations/rep.txt",
        vertex_file=makex_root
        / "DataSets/icews14/processed/original_graph/icews_v.csv",
        relation_file=makex_root / "DataSets/icews14/relation2id.json",
        topk_file=makex_root
        / "local_explanations/output/icews/topk_rep_id_topk1.csv",
        pairs_file=makex_root
        / "local_explanations/output/icews/test_sample_pairs.csv",
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


def resolve_path(raw: str, base_dir: Path) -> Path:
    if not raw:
        return Path()
    path = Path(raw)
    if not path.is_absolute():
        path = (base_dir / raw).resolve()
    return path


def load_vertex_metadata(vertex_path: Path) -> Tuple[Dict[int, str], Dict[int, str]]:
    id_to_name: Dict[int, str] = {}
    label_examples: Dict[int, str] = {}
    if not vertex_path.exists():
        return id_to_name, label_examples
    with vertex_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid_key = row.get("vertex_id:int") or row.get("vertex_id")
            label_key = row.get("label_id:int") or row.get("label_id")
            name = row.get("name:string") or row.get("name") or ""
            if vid_key is None:
                continue
            vid = int(vid_key)
            id_to_name[vid] = name or f"Entity_{vid}"
            if label_key is None:
                continue
            label = int(label_key)
            if label not in label_examples and name:
                label_examples[label] = name
    return id_to_name, label_examples


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


def load_rep_data(rep_path: Path) -> List[dict]:
    reps: List[dict] = []
    if not rep_path.exists():
        return reps
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
            reps.append(
                {
                    "nodes": entry[0],
                    "edges": entry[1],
                    "predicates": entry[2] if len(entry) > 2 else [],
                    "stats": entry[3],
                }
            )
    return reps


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


def describe_label(label_id: int, samples: Dict[int, str]) -> str:
    if label_id in samples:
        return f"标签{label_id}（示例：{samples[label_id]}）"
    return f"标签{label_id}"


def describe_relation(rel_id: int, relation_map: Dict[int, str]) -> str:
    name = relation_map.get(rel_id)
    if not name:
        return f"关系{rel_id}"
    return f"{name}（ID: {rel_id}）"


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


def format_entity(entity_id: int, name_map: Dict[int, str]) -> str:
    name = name_map.get(entity_id, f"实体{entity_id}")
    return f"{name} (ID: {entity_id})"


def main() -> None:
    defaults = default_paths()
    parser = argparse.ArgumentParser(
        description="将 Makex 局部解释结果翻译成中文描述"
    )
    parser.add_argument(
        "--local_file",
        type=Path,
        default=defaults.local_file,
        help="local_topk 日志文件",
    )
    parser.add_argument(
        "--rep_file", type=Path, default=defaults.rep_file, help="rep.txt 路径"
    )
    parser.add_argument(
        "--vertex_file",
        type=Path,
        default=defaults.vertex_file,
        help="顶点 CSV，提供实体名称",
    )
    parser.add_argument(
        "--relation_file",
        type=Path,
        default=defaults.relation_file,
        help="relation2id.json，用于翻译边类型",
    )
    parser.add_argument(
        "--topk_file",
        type=Path,
        default=defaults.topk_file,
        help="topk_rep_id CSV，可由日志推断",
    )
    parser.add_argument(
        "--pairs_file",
        type=Path,
        default=defaults.pairs_file,
        help="test_sample_pairs.csv，可由日志推断",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=5,
        help="最多展示多少条预测解释",
    )
    args = parser.parse_args()

    log_params = parse_log_params(args.local_file)
    base_dir = args.local_file.parent if args.local_file else Path(".")
    if log_params.get("topk_rep_id_file"):
        args.topk_file = resolve_path(log_params["topk_rep_id_file"], base_dir)
    if log_params.get("test_pairs_file"):
        args.pairs_file = resolve_path(log_params["test_pairs_file"], base_dir)

    if not args.topk_file.exists():
        raise SystemExit(f"未找到 topk 文件：{args.topk_file}")

    vertex_names, label_examples = load_vertex_metadata(args.vertex_file)
    relation_map = load_relation_map(args.relation_file)
    rep_data = load_rep_data(args.rep_file)
    pair_map = load_pairs(args.pairs_file)

    entries: "OrderedDict[Tuple[int, int], dict]" = OrderedDict()
    with args.topk_file.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            pair_id = int(row["pair_id"])
            topk_rank = int(row.get("topk", 0))
            key = (pair_id, topk_rank)
            if key in entries:
                continue
            entries[key] = {
                "pair_id": pair_id,
                "topk": topk_rank,
                "pivot_x": int(row["pivot_x"]),
                "pivot_y": int(row["pivot_y"]),
                "rep_id": int(row["rep_id"]),
            }

    print("=== 局部预测解释报告 ===")
    if not entries:
        print("未在 CSV 中找到任何匹配结果。")
        return

    for idx, entry in enumerate(entries.values(), 1):
        if idx > args.max_pairs:
            break
        pair_id = entry["pair_id"]
        user_id, item_id = pair_map.get(pair_id, (entry["pivot_x"], entry["pivot_y"]))
        user_text = format_entity(user_id, vertex_names)
        item_text = format_entity(item_id, vertex_names)
        rep_id = entry["rep_id"]
        rep = rep_data[rep_id] if 0 <= rep_id < len(rep_data) else None
        print(
            f"\n--- 预测 #{idx}：{user_text} -> {item_text} "
            f"(Top-{entry['topk'] + 1}) ---"
        )
        if not rep:
            print("未能在 rep.txt 中找到对应的规则。")
            continue
        stats = rep["stats"]
        support = int(stats[0]) if stats else 0
        conf = stats[1] if len(stats) > 1 else 0.0
        print(
            f"对应规则：Pattern #{rep_id + 1} "
            f"(支持度: {support}, 置信度: {conf * 100:.1f}%)"
        )
        node_info = {int(node[0]): int(node[1]) for node in rep["nodes"]}
        user_edges, item_edges, pivot_x, pivot_y = split_stars(
            rep["edges"], node_info
        )
        print("解释依据：")
        print(
            f"  - 用户星：以 Pattern 节点{pivot_x} 为中心，对应实体 {format_entity(entry['pivot_x'], vertex_names)}"
        )
        for text in describe_edges(
            user_edges, node_info, relation_map, label_examples
        ):
            print(f"      {text}")
        print(
            f"  - 物品星：以 Pattern 节点{pivot_y} 为中心，对应实体 {format_entity(entry['pivot_y'], vertex_names)}"
        )
        for text in describe_edges(
            item_edges, node_info, relation_map, label_examples
        ):
            print(f"      {text}")
        pred_desc = rep.get("predicates") or []
        if pred_desc:
            print("  - 属性限制：")
            for predicate in pred_desc:
                if not predicate:
                    continue
                if predicate[0] == "Constant" and len(predicate) >= 6:
                    _, node_id, attr, value, _, op = predicate[:6]
                    print(
                        f"      节点{node_id} 的 {attr} {op} {value}"
                    )
                else:
                    print(f"      {predicate}")
        else:
            print("  - 属性限制：无")


if __name__ == "__main__":
    main()
