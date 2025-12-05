#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Temporal SARL miner with Transformer policy and rich logging.
Temporal SARL 挖掘器（Miner）模块。
核心功能：
1. 利用训练好的 SARL 策略网络（Policy Network）在时序图上进行有指导的随机游走。
2. 支持正向（Head-Centric）和反向（Tail-Centric）双向挖掘。
3. 使用 Top-K 随机采样策略增加路径多样性。
4. 将挖掘出的具体路径抽象为时空签名（Pattern），并统计高频模式。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from sarl_model import TemporalSARL

# 尝试导入 C++ 扩展 pyMakex，用于高效图查询（如 GetTemporalNeighbors）
# 如果没有编译好，会报错提示
try:
    import pyMakex  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("pyMakex module is required for SARL miner.") from exc


@dataclass(frozen=True)
class TemporalNeighbor:
    """
    定义图中的一条带时间戳的边（邻居）。
    """
    dst: int  # 目标节点 ID
    relation: int  # 关系类型 ID
    timestamp: float  # 发生时间戳


@dataclass
class TemporalPath:
    """
    定义一条挖掘出的时序路径。
    """
    head: int  # 路径的起始节点（可能是 Head 或 Tail，视挖掘方向而定）
    relation: int  # 当前查询试图解释的目标关系
    query_time: float  # 查询发生的时间
    edges: List[TemporalNeighbor]  # 路径上的边序列
    side: str = "head"  # 标记路径方向："head"（用户侧）或 "tail"（物品侧）


@dataclass
class SARLOptions:
    """
    挖掘器的配置参数。
    """
    max_hops: int = 3  # 最大跳数（路径长度）
    history_size: int = 5  # 输送给模型的历史上下文长度
    beam_size: int = 8  # 在每一步搜索时，最多考虑多少个候选邻居（物理截断）
    time_window: float = 30 * 86400.0  # 时间窗口：只考虑最近多久的历史（默认30天）
    min_timestamp: float = 0.0  # 数据集最早时间戳
    time_bucket: float = 7 * 86400.0  # 时间分箱粒度（默认按周）
    device: str = "cpu"  # 运行设备
    log_dir: Path = Path(".")  # 日志保存路径
    verbose: bool = False  # 是否打印详细的每一步日志
    use_amp: bool = False  # 是否开启混合精度推理（仅 GPU 下有效）


class SARLMiner:
    """Run SARL walks on pyMakex graphs with time constraints.
    SARL 挖掘器主类。
    """

    def __init__(
            self,
            model: TemporalSARL,
            options: SARLOptions,
            entity2id_path: Path,
            relation2id_path: Path,
            graph_ptr: int,  # pyMakex C++ 图对象的指针
            edge_store: Dict[int, List[TemporalNeighbor]],  # 正向边索引 Head -> Tail
            rev_edge_store: Optional[Dict[int, List[TemporalNeighbor]]] = None,  # 反向边索引 Tail -> Head
            entity_type_path: Optional[Path] = None,
            entity_type_csv: Optional[Path] = None,
            vertex_file: Optional[Path] = None,
    ) -> None:
        self.model = model.to(options.device)
        self.model.eval()  # 挖掘阶段仅推理，无需反向传播
        self.options = options
        self.graph_ptr = graph_ptr
        self.edge_store = edge_store
        self.rev_edge_store = rev_edge_store or {}

        # 加载 ID 映射，用于日志打印时显示真实名称
        self.entity_map = self._load_map(entity2id_path)
        self.relation_map = self._load_map(relation2id_path)
        self.entity_inv = {v: k for k, v in self.entity_map.items()}
        self.relation_inv = {v: k for k, v in self.relation_map.items()}
        # 加载实体类型映射（优先显式分类 CSV，其次 JSON，最后顶点 CSV）
        self.entity_type_map = self._load_entity_types(entity_type_path, entity_type_csv, vertex_file)

        # 定义填充 ID（通常是最大 ID + 1）
        self.pad_entity = len(self.entity_map)
        self.pad_relation = len(self.relation_map)
        self.time_bucket = max(0.0, self.options.time_bucket)

        # 初始化日志目录
        self.log_dir = options.log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.raw_path_file = self.log_dir / "sarl_raw_paths.txt"
        self.raw_path_file.write_text("")  # 清空旧日志
        self.reset_statistics()

    def reset_statistics(self) -> None:
        """重置统计计数器。"""
        self.stats_attempted = 0
        self.stats_hits = 0

    @staticmethod
    def _load_map(path: Path) -> Dict[str, int]:
        if not path.exists():
            raise FileNotFoundError(f"Mapping file not found: {path}")
        return json.loads(path.read_text())

    def _id_to_name(self, inv_map: Dict[int, str], idx: int) -> str:
        return inv_map.get(idx, f"ID_{idx}")

    def _load_entity_types(
            self,
            entity_type_path: Optional[Path],
            entity_type_csv: Optional[Path],
            vertex_file: Optional[Path],
    ) -> Dict[int, str]:
        """
        加载实体类型映射：
        1) 若提供 entity_classification.csv（含 Entity_ID / Type_Name），优先使用。
        2) 其次尝试 entity2type.json。
        3) 最后回退到顶点 CSV 的 type 列。
        """
        mapping: Dict[int, str] = {}
        # 1) 显式分类 CSV
        if entity_type_csv and entity_type_csv.exists():
            try:
                with entity_type_csv.open("r", encoding="utf-8", errors="replace") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        lower_row = {k.lower(): v for k, v in row.items() if k}
                        vid_raw = lower_row.get("entity_id")
                        type_name = lower_row.get("type_name") or lower_row.get("type")
                        if vid_raw is None or type_name is None:
                            continue
                        try:
                            ent_id = int(vid_raw)
                        except ValueError:
                            continue
                        mapping[ent_id] = str(type_name)
            except Exception as exc:  # pragma: no cover - 容错输出提醒
                print(f"[WARN] 加载实体分类 CSV 失败 {entity_type_csv}: {exc}")
        if mapping:
            return mapping

        # 2) JSON 映射
        if entity_type_path and entity_type_path.exists():
            try:
                raw = json.loads(entity_type_path.read_text())
                for key, value in raw.items():
                    ent_id: Optional[int] = None
                    try:
                        ent_id = int(key)
                    except ValueError:
                        ent_id = self.entity_map.get(key)
                    if ent_id is None:
                        continue
                    mapping[ent_id] = str(value)
            except Exception as exc:  # pragma: no cover - 容错输出提醒
                print(f"[WARN] 加载实体类型文件失败 {entity_type_path}: {exc}")

        if not mapping and vertex_file and vertex_file.exists():
            try:
                with vertex_file.open("r", encoding="utf-8", errors="replace") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        vid_raw = row.get("vertex_id:int") or row.get("vertex_id")
                        type_raw = row.get("type:string") or row.get("type")
                        if vid_raw is None or type_raw is None:
                            continue
                        try:
                            mapping[int(vid_raw)] = str(type_raw)
                        except ValueError:
                            continue
            except Exception as exc:  # pragma: no cover - 容错输出提醒
                print(f"[WARN] 读取顶点类型列失败 {vertex_file}: {exc}")
        return mapping

    def _entity_type(self, entity_id: int) -> str:
        """
        返回实体的类型名称；若类型缺失则回退到实体原始名称，确保输出可读。
        """
        if entity_id in self.entity_type_map:
            return self.entity_type_map[entity_id]
        return self.entity_inv.get(entity_id, f"Entity_{entity_id}")

    def _relation_name(self, relation_id: int) -> str:
        """关系名称转为可读格式（替换下划线）。"""
        raw = self.relation_inv.get(relation_id, f"rel_{relation_id}")
        return raw.replace("_", " ")

    def _time_bin(self, query_time: float, edge_ts: float) -> Tuple[int, float]:
        """
        计算时间分箱编号以及时间差（秒）。
        """
        delta = max(0.0, query_time - edge_ts)
        if self.time_bucket <= 0:
            return 0, delta
        return int(delta // self.time_bucket), delta

    def mine_paths(
            self,
            head_id: int,
            relation_id: int,
            query_time: float,
            num_walks: int,
    ) -> List[TemporalPath]:
        """
        正向挖掘入口：从 Head 出发，寻找解释路径。
        对应 Makex 中的 User Star 挖掘。
        """
        results: List[TemporalPath] = []
        for walk_idx in range(num_walks):
            self.stats_attempted += 1
            # 调用单次游走，指定 neighbor_getter 为正向邻居查找
            path = self._single_walk(
                pivot_id=head_id,
                relation_id=relation_id,
                query_time=query_time,
                walk_idx=walk_idx,
                neighbor_getter=self._temporal_neighbors,
                side="head",
            )
            if path:
                self.stats_hits += 1
                results.append(path)
        return results

    def mine_reverse_paths(
            self,
            tail_id: int,
            relation_id: int,
            query_time: float,
            num_walks: int,
    ) -> List[TemporalPath]:
        """
        反向挖掘入口：从 Tail 出发，寻找解释路径。
        对应 Makex 中的 Item Star 挖掘。
        注意：这里是在时间轴上回溯 Tail 的历史，寻找“为什么 Tail 会被选中”。
        """
        results: List[TemporalPath] = []
        for walk_idx in range(num_walks):
            self.stats_attempted += 1
            # 调用单次游走，指定 neighbor_getter 为反向邻居查找
            path = self._single_walk(
                pivot_id=tail_id,
                relation_id=relation_id,
                query_time=query_time,
                walk_idx=walk_idx,
                neighbor_getter=self._temporal_reverse_neighbors,
                side="tail",
            )
            if path:
                self.stats_hits += 1
                results.append(path)
        return results

    def _single_walk(
            self,
            pivot_id: int,
            relation_id: int,
            query_time: float,
            walk_idx: int,
            neighbor_getter,
            side: str,
    ) -> Optional[TemporalPath]:
        """
        执行单次随机游走的核心逻辑。
        :param pivot_id: 起始节点 ID (Head or Tail)
        :param neighbor_getter: 获取邻居的函数（正向或反向）
        """
        # 1. 初始化历史记忆（用 PAD 填充）
        history_entities, history_relations, history_deltas = self._init_history(pivot_id, relation_id)

        current = pivot_id
        current_time = query_time
        mined_edges: List[TemporalNeighbor] = []

        # 2. 开始逐跳搜索
        for hop in range(self.options.max_hops):
            # 获取当前节点在当前时间之前的邻居
            neighbors = neighbor_getter(current, current_time)
            if not neighbors:
                # 如果是 verbose 模式，打印死胡同信息
                if self.options.verbose:
                    print(f"[SARL Step] Walk {walk_idx}, hop {hop}: no neighbors in window.")
                return None

            # 3. 询问模型，选择下一步
            choice, probs = self._select_neighbor(
                neighbors,
                history_entities,
                history_relations,
                history_deltas,
                current,
                relation_id,
                query_time,
            )

            mined_edges.append(choice)

            # 记录日志（仅 verbose=True 时）
            self._log_step(
                walk_idx,
                hop,
                current,
                relation_id,
                current_time,
                neighbors,
                probs,
                choice,
            )

            # 4. 更新历史记忆（Sliding Window）
            # 将刚走过的边加入历史序列，挤出最旧的
            self._append_history(
                history_entities,
                history_relations,
                history_deltas,
                choice.dst,
                choice.relation,
                max(0.0, query_time - choice.timestamp),
            )

            # 5. 移动到下一个节点，时间回溯
            current = choice.dst
            current_time = choice.timestamp

            # 达到最大长度，停止
            if len(mined_edges) == self.options.max_hops:
                break

        # 6. 成功找到路径，保存并返回
        if mined_edges:
            self._write_raw_path(pivot_id, relation_id, query_time, mined_edges, side)
            return TemporalPath(head=pivot_id, relation=relation_id, query_time=query_time, edges=mined_edges,
                                side=side)
        return None

    def _init_history(self, head: int, relation: int) -> Tuple[List[int], List[int], List[float]]:
        """初始化长度为 history_size 的空白历史序列。"""
        entities = [head]
        relations = [relation]
        deltas = [0.0]
        while len(entities) < self.options.history_size:
            entities.append(self.pad_entity)
            relations.append(self.pad_relation)
            deltas.append(0.0)
        return entities, relations, deltas

    def _append_history(
            self,
            entities: List[int],
            relations: List[int],
            deltas: List[float],
            entity: int,
            relation: int,
            delta: float,
    ) -> None:
        """更新历史序列（FIFO 队列）。"""
        entities.append(entity)
        relations.append(relation)
        deltas.append(delta)
        if len(entities) > self.options.history_size:
            entities.pop(0)
            relations.pop(0)
            deltas.pop(0)

    def _history_tensors(
            self,
            entities: List[int],
            relations: List[int],
            deltas: List[float],
            current_entity: int,
            relation_id: int,
    ) -> Tuple[torch.Tensor, ...]:
        """将 Python 列表转换为 PyTorch Tensor，准备输入模型。"""
        device = self.options.device
        hist_entities = torch.tensor([entities], dtype=torch.long, device=device)
        hist_relations = torch.tensor([relations], dtype=torch.long, device=device)
        hist_deltas = torch.tensor([deltas], dtype=torch.float32, device=device)
        current_tensor = torch.tensor([current_entity], dtype=torch.long, device=device)
        relation_tensor = torch.tensor([relation_id], dtype=torch.long, device=device)
        return hist_entities, hist_relations, hist_deltas, current_tensor, relation_tensor

    def _select_neighbor(
            self,
            neighbors: List[TemporalNeighbor],
            history_entities: List[int],
            history_relations: List[int],
            history_deltas: List[float],
            current_entity: int,
            relation_id: int,
            query_time: float,
    ) -> Tuple[TemporalNeighbor, torch.Tensor]:
        """
        核心决策函数：Top-K 随机采样。
        """
        # 1. 准备候选邻居的 Tensor 数据
        cand_entities = torch.tensor([[n.dst for n in neighbors]], dtype=torch.long, device=self.options.device)
        cand_relations = torch.tensor([[n.relation for n in neighbors]], dtype=torch.long, device=self.options.device)
        cand_deltas = torch.tensor(
            [[max(0.0, query_time - n.timestamp) for n in neighbors]],
            dtype=torch.float32,
            device=self.options.device,
        )

        # 2. 准备历史 Context Tensor
        hist_entities, hist_relations, hist_deltas, current_tensor, relation_tensor = self._history_tensors(
            history_entities,
            history_relations,
            history_deltas,
            current_entity,
            relation_id,
        )

        # 3. 模型推理 (Forward)
        self.model.eval()
        with torch.no_grad():
            if self.options.use_amp and torch.cuda.is_available():
                autocast_ctx = torch.cuda.amp.autocast()
            else:
                from contextlib import nullcontext
                autocast_ctx = nullcontext()
            with autocast_ctx:
                scores = self.model(
                    hist_entities,
                    hist_relations,
                    hist_deltas,
                    current_tensor,
                    relation_tensor,
                    cand_entities,
                    cand_relations,
                    cand_deltas,
                )
        # 避免出现 NaN/Inf
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        # 4. 计算概率分布 (Softmax)
        probs = torch.softmax(scores.squeeze(0), dim=-1)
        if (not torch.isfinite(probs).all()) or probs.sum() <= 0:
            probs = torch.full_like(probs, 1.0 / len(probs))

        # 5. Top-K 采样逻辑
        # 选取前 5 个（如果邻居不够5个，就全选）
        k = min(5, len(neighbors))
        if k <= 0:
            raise RuntimeError("No neighbors available for selection.")

        # 获取 Top-K 的概率值和原始索引
        top_probs, top_idx = torch.topk(probs, k=k)

        # 重新归一化 (Re-normalize)，让这K个概率加起来等于1
        normalized = torch.softmax(top_probs, dim=-1)
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        if (not torch.isfinite(normalized).all()) or normalized.sum() <= 0:
            normalized = torch.full_like(normalized, 1.0 / len(normalized))

        # 在这K个里面随机抽一个 (Multinomial Sampling)
        sampled = torch.multinomial(normalized, num_samples=1).item()

        # 映射回原始邻居列表
        selected_neighbor = neighbors[top_idx[sampled].item()]

        # 构造返回的概率分布（只保留被选中的那几个，方便调试）
        sampled_probs = torch.zeros_like(probs)
        sampled_probs[top_idx] = normalized

        return selected_neighbor, sampled_probs

    def _temporal_neighbors(self, node_id: int, time_upper: float) -> List[TemporalNeighbor]:
        """正向邻居查询：查 Head -> Tail。"""
        ts_upper = int(time_upper)
        ts_lower = int(max(self.options.min_timestamp, time_upper - self.options.time_window))

        # 尝试使用 pyMakex C++ 加速查询
        candidate_ids: Optional[Iterable[int]] = None
        try:
            candidate_ids = pyMakex.GetTemporalNeighbors(
                self.graph_ptr, int(node_id), ts_lower, ts_upper, 1
            )
        except TypeError:
            # 兼容旧版本接口
            candidate_ids = pyMakex.GetTemporalNeighbors(int(node_id), ts_lower, ts_upper)

        neighbors: List[TemporalNeighbor] = []
        allowed = set(candidate_ids) if candidate_ids else None

        # 从 Python 字典中筛选符合条件的边
        for edge in self.edge_store.get(node_id, []):
            # 时间窗口过滤
            if not (ts_lower <= edge.timestamp <= ts_upper):
                continue
            # 确保该邻居也在 C++ 查询结果中（双重验证，可选）
            if allowed is not None and edge.dst not in allowed:
                continue
            neighbors.append(edge)

        # 按时间倒序排列
        neighbors.sort(key=lambda e: e.timestamp, reverse=True)

        # Beam Search 截断：只取最近的 beam_size 个
        if len(neighbors) > self.options.beam_size:
            neighbors = neighbors[: self.options.beam_size]
        return neighbors

    def _temporal_reverse_neighbors(self, node_id: int, time_upper: float) -> List[TemporalNeighbor]:
        """反向邻居查询：查 Tail -> Head（查询 rev_edge_store）。"""
        ts_upper = int(time_upper)
        ts_lower = int(max(self.options.min_timestamp, time_upper - self.options.time_window))
        neighbors: List[TemporalNeighbor] = []

        for edge in self.rev_edge_store.get(node_id, []):
            if ts_lower <= edge.timestamp <= ts_upper:
                neighbors.append(edge)

        neighbors.sort(key=lambda e: e.timestamp, reverse=True)

        if len(neighbors) > self.options.beam_size:
            neighbors = neighbors[: self.options.beam_size]
        return neighbors

    def _log_step(
            self,
            walk_idx: int,
            hop: int,
            current: int,
            goal_relation: int,
            current_time: float,
            neighbors: List[TemporalNeighbor],
            probs: torch.Tensor,
            selected: TemporalNeighbor,
    ) -> None:
        """打印单步决策日志（Verbose 模式下）。"""
        if not self.options.verbose:
            return
        entity_name = self._id_to_name(self.entity_inv, current)
        goal_name = self._id_to_name(self.relation_inv, goal_relation)
        current_time_str = self._format_ts(current_time)
        print(
            f"[SARL Step] Walk {walk_idx}, hop {hop}, Current: {entity_name} (t={current_time_str}), Goal Rel: {goal_name}")
        sorted_idx = torch.argsort(probs, descending=True)
        top = sorted_idx[: min(3, len(sorted_idx))]
        top_desc = ", ".join(
            f"{self._id_to_name(self.relation_inv, neighbors[i].relation)}({probs[i].item():.2f})" for i in top
        )
        low_idx = sorted_idx[-1].item()
        low_desc = f"{self._id_to_name(self.relation_inv, neighbors[low_idx].relation)}({probs[low_idx].item():.2f})"
        print(f"  [Model Decision] High scores for: {top_desc}")
        print(f"  [Model Decision] Low scores for: {low_desc}")
        print(
            f"  [Action] Selected {self._id_to_name(self.relation_inv, selected.relation)}"
            f" -> {self._id_to_name(self.entity_inv, selected.dst)}"
            f" at {self._format_ts(selected.timestamp)}"
        )

    def _write_raw_path(
            self,
            head: int,
            relation: int,
            query_time: float,
            edges: List[TemporalNeighbor],
            side: str,
    ) -> None:
        """将挖掘出的完整路径写入日志文件。"""
        human_edges = " -> ".join(
            f"{self._id_to_name(self.relation_inv, edge.relation)}"
            f"({self._format_ts(edge.timestamp)}) => {self._id_to_name(self.entity_inv, edge.dst)}"
            for edge in edges
        )
        # 标记是 Head 侧还是 Tail 侧
        with self.raw_path_file.open("a", encoding="utf-8") as fp:
            fp.write(
                f"Query[{side}]({self._id_to_name(self.entity_inv, head)}, {self._id_to_name(self.relation_inv, relation)}"
                f" @ {self._format_ts(query_time)}) -> Path: {human_edges}\n"
            )

    def report_performance(self) -> None:
        """任务结束后打印总体统计数据。"""
        attempted = max(1, self.stats_attempted)
        hit_rate = self.stats_hits / attempted * 100
        print(
            "[SARL Performance]\n"
            f"- Total Walks Attempted: {self.stats_attempted}\n"
            f"- Valid Paths Found (Hits): {self.stats_hits}\n"
            f"- Hit Rate: {hit_rate:.2f}% (Benchmark: Random Walk < 1%)"
        )

    # 注意：此处函数名 t 似乎是笔误，应为 cluster_paths，这里保持原样但加上正确注释逻辑
    def cluster_paths(
            self, paths: List[TemporalPath], time_bucket: float
    ) -> Dict[Tuple[int, int, str, str], List[TemporalPath]]:
        """
        路径聚类函数。
        将结构相同、时间模式相同的路径归为一类。
        """
        grouped: Dict[Tuple[int, int, str, str], List[TemporalPath]] = {}
        for path in paths:
            signature = self._build_signature(path, time_bucket)
            # Key: (Head, Rel, Signature, Side) -> 区分了 Head 侧和 Tail 侧规则
            key = (path.head, path.relation, signature, path.side)
            grouped.setdefault(key, []).append(path)
        return grouped

    def _build_signature(self, path: TemporalPath, bucket: float) -> str:
        """
        生成路径的时空签名 (Pattern Signature)。
        格式示例: L3|Visit:0|Support:1
        """
        tokens = [f"L{len(path.edges)}"]
        for edge in path.edges:
            # 计算时间差
            delta = max(0.0, path.query_time - edge.timestamp)
            # 时间分箱 + 关系 ID
            tokens.append(f"{edge.relation}:{int(delta // bucket)}")
        return "|".join(tokens)

    def path_to_rep(self, path: TemporalPath, support: int) -> List:
        """
        将一条代表路径转换为精简可读的规则格式。
        输出结构：
        {
          "nodes": [{"pattern_id":1,"entity_id":123,"name":"China","type":"Country"}, ...],
          "edges": [{"src":1,"dst":2,"relation_id":53,"relation":"Threaten","time_bin":0,"gap_days":0.0}, ...],
          "query": {"relation_id":53,"relation":"Threaten","path_length":3,"star_side":"head"},
          "support": float,
          "confidence": 1.0
        }
        """
        # 1. 节点重映射：将具体 ID 转换为抽象 ID (1, 2, 3...) 并附带类型、名称、原始 ID
        node_ids: Dict[int, int] = {path.head: 1}
        node_info: Dict[int, Dict[str, str]] = {
            1: {
                "entity_id": path.head,
                "name": self._id_to_name(self.entity_inv, path.head),
                "type": self._entity_type(path.head),
            }
        }
        next_idx = 2
        current = path.head
        edges_payload: List[Dict[str, object]] = []

        for hop_idx, edge in enumerate(path.edges, 1):
            if edge.dst not in node_ids:
                node_ids[edge.dst] = next_idx
                node_info[next_idx] = {
                    "entity_id": edge.dst,
                    "name": self._id_to_name(self.entity_inv, edge.dst),
                    "type": self._entity_type(edge.dst),
                }
                next_idx += 1
            src_idx = node_ids[current]
            dst_idx = node_ids[edge.dst]

            bucket, delta = self._time_bin(path.query_time, edge.timestamp)
            edges_payload.append(
                {
                    "src": src_idx,
                    "dst": dst_idx,
                    "relation_id": edge.relation,
                    "relation": self._relation_name(edge.relation),
                    "time_bin": bucket,
                    "gap_days": round(delta / 86400.0, 2),
                }
            )

            current = edge.dst

        # 2. 构建节点列表（包含原始 ID、名称、类型）
        vertices = [
            {
                "pattern_id": idx,
                "entity_id": info["entity_id"],
                "name": info["name"],
                "type": info["type"],
            }
            for idx, info in sorted(node_info.items(), key=lambda kv: kv[0])
        ]

        # 3. 查询信息与统计
        rep_entry = {
            "nodes": vertices,
            "edges": edges_payload,
            "query": {
                "relation_id": path.relation,
                "relation": self._relation_name(path.relation),
                "path_length": len(path.edges),
                "star_side": path.side,
            },
            "support": float(support),
            "confidence": 1.0,
        }
        return rep_entry

    @staticmethod
    def _format_ts(ts: float) -> str:
        """辅助函数：将时间戳格式化为日期字符串。"""
        try:
            return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
        except (ValueError, OSError, OverflowError):
            return f"{ts:.2f}"


__all__ = [
    "SARLMiner",
    "SARLOptions",
    "TemporalNeighbor",
    "TemporalPath",
]
