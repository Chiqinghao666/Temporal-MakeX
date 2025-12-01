#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将训练好的 TemporalSARL 模型导出为 TorchScript，供 C++ (LibTorch) 载入。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sarl_model import TemporalSARL


def export(model_path: Path, output_path: Path, num_entities: int, num_relations: int,
           history_size: int, num_candidates: int) -> None:
    # 构建模型并加载权重
    base_model = TemporalSARL(num_entities=num_entities, num_relations=num_relations)
    state_dict = torch.load(model_path, map_location="cpu")
    base_model.load_state_dict(state_dict)
    base_model.eval()

    # 使用包装器避免重复定义 forward 的冲突
    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self,
                    hist_entities,
                    hist_relations,
                    hist_deltas,
                    current_entities,
                    query_relation,
                    cand_entities,
                    cand_relations,
                    cand_deltas):
            return self.m(hist_entities,
                          hist_relations,
                          hist_deltas,
                          current_entities,
                          query_relation,
                          cand_entities,
                          cand_relations,
                          cand_deltas)

    model = Wrapper(base_model)

    # 准备 Dummy 输入，维度需与 forward 对齐
    # (Batch, Seq) 历史
    hist_entities = torch.zeros((1, history_size), dtype=torch.long)
    hist_relations = torch.zeros((1, history_size), dtype=torch.long)
    hist_deltas = torch.zeros((1, history_size), dtype=torch.float32)

    # (Batch,) 当前节点与查询关系
    current_entities = torch.zeros((1,), dtype=torch.long)
    query_relation = torch.zeros((1,), dtype=torch.long)

    # (Batch, Num_Candidates)
    cand_entities = torch.zeros((1, num_candidates), dtype=torch.long)
    cand_relations = torch.zeros((1, num_candidates), dtype=torch.long)
    cand_deltas = torch.zeros((1, num_candidates), dtype=torch.float32)

    example_inputs = (
        hist_entities,
        hist_relations,
        hist_deltas,
        current_entities,
        query_relation,
        cand_entities,
        cand_relations,
        cand_deltas,
    )

    # trace 导出
    with torch.no_grad():
        traced = torch.jit.trace(model, example_inputs, strict=False)
    traced.save(output_path)
    print(f"[OK] 导出完成: {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export TemporalSARL to TorchScript")
    p.add_argument("--model_path", type=Path, default=Path("./sarl_model.pth"))
    p.add_argument("--output_path", type=Path, default=Path("./sarl_model_traced.pt"))
    p.add_argument("--num_entities", type=int, required=True)
    p.add_argument("--num_relations", type=int, required=True)
    p.add_argument("--history_size", type=int, default=8)
    p.add_argument("--num_candidates", type=int, default=16)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export(
        model_path=args.model_path,
        output_path=args.output_path,
        num_entities=args.num_entities,
        num_relations=args.num_relations,
        history_size=args.history_size,
        num_candidates=args.num_candidates,
    )
