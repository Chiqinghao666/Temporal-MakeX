#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main entry for Temporal-SARL discovery (mining).
Temporal-SARL 规则挖掘主程序。
核心流程：
1. 加载数据和预训练模型。
2. 构建双向图索引（正向+反向）。
3. 采样测试查询（Query）。
4. 执行双向路径挖掘（Head侧 + Tail侧）。
5. 对挖掘出的路径进行抽象、聚类，生成 REP 规则文件。
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

# 尝试导入 C++ 扩展库，用于高效图计算（可选）
try:
    import pyMakex
    print("[INFO] 已成功导入 pyMakex（C++ 扩展已可用）")
except ImportError:
    pyMakex = None
    print("[WARN] 未找到 pyMakex（将退回纯 Python 路径挖掘）")

from sarl_model import TemporalSARL
from temporal_sarl_miner import SARLMiner, SARLOptions, TemporalNeighbor, TemporalPath

# 定义三元组类型：(head_id, relation_id, tail_id, timestamp)
Triple = Tuple[int, int, int, float]


def load_json(path: Path) -> Dict[str, int]:
    """
    加载 ID 映射文件（如 entity2id.json）。
    """
    return json.loads(path.read_text())


def parse_temporal_line(
        line: str,
        entity_map: Dict[str, int],
        relation_map: Dict[str, int],
) -> Tuple[int, int, int, float] | None:
    """
    解析一行原始文本数据。
    格式示例："China\tVisit\tJapan\t2014-01-01"
    输出：(head_id, rel_id, tail_id, timestamp)
    """
    parts = line.strip().split("\t")
    if len(parts) < 4:
        return None
    head, relation, tail, date_str = parts[:4]

    # 过滤掉未在字典中出现的实体/关系
    if head not in entity_map or tail not in entity_map or relation not in relation_map:
        return None
    try:
        # 转换日期字符串为时间戳
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
    """
    从 train/valid/test 文件中加载所有时序三元组。
    :param limit: 可选的采样数量限制，用于快速测试。
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
                if limit and len(triples) >= limit:
                    return triples
    return triples


def build_edge_store(
        triples: Sequence[Triple],
) -> Tuple[Dict[int, List[TemporalNeighbor]], Dict[int, List[TemporalNeighbor]]]:
    """
    构建内存中的双向图索引。
    输出两个字典：
    1. store (正向索引): Head -> [(Tail, Rel, Time), ...]
    2. rev_store (反向索引): Tail -> [(Head, Rel, Time), ...]
    关键点：每个列表都按时间倒序排列，方便快速检索最近历史。
    """
    store: Dict[int, List[TemporalNeighbor]] = {}
    rev_store: Dict[int, List[TemporalNeighbor]] = {}

    for head, relation, tail, ts in triples:
        # 正向边：Head -> Tail
        store.setdefault(head, []).append(TemporalNeighbor(dst=tail, relation=relation, timestamp=ts))
        # 反向边：Tail -> Head
        rev_store.setdefault(tail, []).append(TemporalNeighbor(dst=head, relation=relation, timestamp=ts))

    # 排序优化查询效率
    for edges in store.values():
        edges.sort(key=lambda e: e.timestamp, reverse=True)
    for edges in rev_store.values():
        edges.sort(key=lambda e: e.timestamp, reverse=True)

    return store, rev_store


def sample_queries(triples: Sequence[Triple], num_queries: int, start: int = 0) -> List[Triple]:
    """
    随机采样一部分真实发生的三元组作为“查询（Query）”。
    这是为了模拟链接预测任务：已知 (h, r, ?, t)，预测 t。
    :param start: 从打乱后的列表起始位置开始取（用于多进程切片）
    """
    triples = list(triples)
    random.shuffle(triples)
    if start >= len(triples):
        return []
    end = min(len(triples), start + num_queries)
    return triples[start:end]


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    包含路径配置、模型参数、挖掘超参数（如 max_hops, beam_size）等。
    """
    parser = argparse.ArgumentParser(description="Execute Temporal-SARL mining on ICEWS.")
    # 数据集相关路径
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

    # 模型路径
    parser.add_argument("--model_path", type=Path, default=Path("./sarl_model.pth"))

    # 挖掘参数
    parser.add_argument("--num_queries", type=int, default=200, help="采样多少个查询进行挖掘")
    parser.add_argument("--query_start", type=int, default=0, help="采样起始偏移（多进程切片用）")
    parser.add_argument("--walks_per_query", type=int, default=50, help="每个查询采样多少条路径")
    parser.add_argument("--max_hops", type=int, default=3, help="最大路径长度")
    parser.add_argument("--beam_size", type=int, default=16, help="集束搜索的宽度")
    parser.add_argument("--history_size", type=int, default=8, help="模型输入的历史序列长度")
    parser.add_argument("--progress_interval", type=int, default=20, help="每多少个查询打印一次 ETA")

    # 时间相关参数
    parser.add_argument("--time_window", type=float, default=30 * 86400.0, help="回溯历史的时间窗口（秒）")
    parser.add_argument("--time_bucket", type=float, default=7 * 86400.0, help="生成签名时的时间分箱粒度（秒）")

    # 输出控制
    parser.add_argument("--top_signatures", type=int, default=200, help="保留多少个高频模式")
    parser.add_argument("--sample_limit", type=int, default=0)
    parser.add_argument("--output_rep", type=Path, default=Path("./rep_sarl.txt"))
    parser.add_argument("--log_dir", type=Path, default=Path("./global_explanations"))
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=1996)
    parser.add_argument("--verbose", action="store_true", help="是否打印每一步的详细日志")
    return parser.parse_args()


def main() -> None:
    """
    主程序入口。
    """
    args = parse_args()

    # 1. 初始化随机种子，保证可复现性
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 2. 加载 ID 映射和原始数据
    entity_map = load_json(args.entity_map)
    relation_map = load_json(args.relation_map)
    triples = load_temporal_triples(args.dataset_dir, entity_map, relation_map, args.sample_limit or None)
    if not triples:
        raise RuntimeError("No temporal triples found; ensure ICEWS train/valid/test exist.")

    # 3. 构建图索引和采样查询
    edge_store, rev_edge_store = build_edge_store(triples)
    queries = sample_queries(triples, args.num_queries, args.query_start)
    print(f"[Init] Loaded {len(triples)} triples, sampled {len(queries)} queries (start={args.query_start}).")

    # 4. 加载预训练好的 TemporalSARL 模型
    num_entities = len(entity_map) + 1
    num_relations = len(relation_map) + 1
    model = TemporalSARL(num_entities=num_entities, num_relations=num_relations)
    state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict)

    # 5. 初始化挖掘器 (Miner)
    # 尝试加载 C++ 图指针，如果失败则使用纯 Python
    if pyMakex:
        graph_ptr = pyMakex.ReadDataGraph(str(args.vertex_file), str(args.edge_file))
    else:
        graph_ptr = 0

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    options = SARLOptions(
        max_hops=args.max_hops,
        history_size=args.history_size,
        beam_size=args.beam_size,
        time_window=args.time_window,
        device=device,
        log_dir=args.log_dir,
        verbose=args.verbose,
        use_amp=True if device == "cuda" else False,
    )

    miner = SARLMiner(
        model=model,
        options=options,
        entity2id_path=args.entity_map,
        relation2id_path=args.relation_map,
        graph_ptr=graph_ptr,
        edge_store=edge_store,
        rev_edge_store=rev_edge_store,  # 传入反向索引，支持双向挖掘
    )
    miner.reset_statistics()

    # 6. 开始挖掘循环
    all_paths: List[TemporalPath] = []
    total_queries = len(queries)

    # 计时器设置
    import time
    wall_start = time.time()

    progress_interval = max(1, args.progress_interval)

    for idx, (head, relation, tail, ts) in enumerate(queries, 1):
        # --- 双向挖掘核心 ---
        # A. 从 Head 出发挖掘（User Star）
        head_paths = miner.mine_paths(head, relation, ts, num_walks=args.walks_per_query)

        # B. 从 Tail 出发反向挖掘（Item Star）
        tail_paths = miner.mine_reverse_paths(tail, relation, ts, num_walks=args.walks_per_query)

        # 收集结果
        all_paths.extend(head_paths)
        all_paths.extend(tail_paths)

        # 打印进度和预计剩余时间 (ETA)
        elapsed = time.time() - wall_start
        if idx % progress_interval == 0 or idx == total_queries:
            avg_per_query = elapsed / idx
            remaining = avg_per_query * (total_queries - idx)
            qps = idx / max(1e-6, elapsed)
            print(
                f"[Progress] {idx}/{total_queries} | {elapsed/60:.1f}m elapsed | ETA {remaining/60:.1f}m | {qps:.2f} q/s"
            )

    miner.report_performance()
    if not all_paths:
        print("[Warning] No paths mined; rep file will not be created.")
        return

    # 7. 路径聚类与模式抽象
    # 将挖掘出的海量具体路径，根据结构和时间签名进行聚类
    grouped = miner.cluster_paths(all_paths, args.time_bucket)

    # 按支持度（出现频次）排序，保留 Top-K
    ranked = sorted(grouped.items(), key=lambda kv: len(kv[1]), reverse=True)

    # 8. 格式转换并保存
    rep_entries = []
    # 解包 key: (head, relation, signature, side)
    for (head, relation, _, side), path_group in ranked[: args.top_signatures]:
        # 将聚类后的代表路径转换为 Makex 兼容的规则格式
        rep_entries.append(miner.path_to_rep(path_group[0], len(path_group)))

    # 写入 rep_sarl.txt
    with args.output_rep.open("w", encoding="utf-8") as fp:
        for entry in rep_entries:
            fp.write(f"{entry}\n")

    print(
        f"[Summary] Saved {len(rep_entries)} SARL patterns to {args.output_rep}."
        f" Raw paths logged at {options.log_dir / 'sarl_raw_paths.txt'}"
    )


if __name__ == "__main__":
    main()
