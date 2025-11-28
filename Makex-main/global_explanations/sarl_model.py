#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Temporal-SARL transformer policy network.
Temporal-SARL 的 Transformer 策略网络模型定义。
核心功能：利用 Transformer 编码历史交互序列，结合时间信息，预测下一步可能的路径/链接。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    """
    Continuous-time encoding combining linear and periodic terms.
    时间编码层：将连续的时间数值（时间差）转换为向量表示。
    原理参考 Time2Vec 论文，结合了线性项（捕捉趋势）和周期项（捕捉正弦规律）。
    """

    def __init__(self, dim: int) -> None:
        """
        初始化 Time2Vec 层。
        :param dim: 输出的时间向量维度，必须 >= 2。
        """
        super().__init__()
        if dim < 2:
            raise ValueError("Time2Vec dimension must be >= 2")
        # 线性部分的权重和偏置 (Linear term)
        self.linear_weight = nn.Parameter(torch.randn(1))
        self.linear_bias = nn.Parameter(torch.zeros(1))
        # 周期部分的频率和相位 (Periodic term: sin(wt + phi))
        self.freq = nn.Parameter(torch.randn(dim - 1))
        self.phase = nn.Parameter(torch.zeros(dim - 1))

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        :param delta: 输入的时间差张量，形状通常为 (Batch, ...)
        :return: 编码后的时间向量，形状为 (Batch, ..., dim)
        """
        # 扩展维度以便进行广播计算: (Batch, ..., 1)
        delta = delta.unsqueeze(-1)

        # 计算线性部分: w * t + b
        linear = self.linear_weight * delta + self.linear_bias

        # 计算周期部分: sin(w * t + phi)
        # view(1, 1, -1) 是为了广播到前面的 batch 和 seq 维度
        sinusoid = torch.sin(delta * self.freq.view(1, 1, -1) + self.phase.view(1, 1, -1))

        # 拼接两部分：[线性(1维), 周期(dim-1维)] -> 总共 dim 维
        return torch.cat([linear, sinusoid], dim=-1)


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    标准 Transformer 位置编码。
    作用：因为 Transformer 是并行处理序列的，需要加入位置信息让模型知道事件发生的先后顺序。
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 预计算位置编码矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算分母中的 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 偶数维度用 sin，奇数维度用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加 batch 维度: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # 注册为 buffer，这样它会随模型保存，但不会被视为可训练参数（梯度更新）
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        :param x: 输入的嵌入序列，形状 (Batch, Seq_Len, Dim)
        """
        # 将位置编码加到输入上（截取与输入序列长度对应的部分）
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TemporalSARL(nn.Module):
    """
    Temporal-aware transformer scoring module.
    核心模型类：基于时序感知的 Transformer 评分模型。
    """

    def __init__( 
            self,
            num_entities: int,
            num_relations: int,
            embed_dim: int = 128,
            nhead: int = 4,
            num_layers: int = 2,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # 1. 定义嵌入层 (Embedding Layers)
        self.entity_emb = nn.Embedding(num_entities, embed_dim)
        self.relation_emb = nn.Embedding(num_relations, embed_dim)

        # 2. 时间编码器
        self.time_enc = Time2Vec(embed_dim)

        # 3. 位置编码器
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)

        # 4. Transformer 编码器
        # batch_first=False 表示输入格式默认为 (Seq_Len, Batch, Dim)，这是 PyTorch Transformer 的默认习惯
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dropout=dropout, batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5. 状态投影层：把拼接后的长向量压缩回 embed_dim
        # 输入维度是 4 * embed_dim (Context + Current_Entity + Query_Rel + Last_Time)
        self.state_proj = nn.Linear(embed_dim * 4, embed_dim)

        # 6. 策略头 (Policy Head)：最后的打分网络 (MLP)
        self.policy_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),  # 输出一个标量分数
        )
        self.dropout = nn.Dropout(dropout)

        # 初始化参数
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """使用 Xavier 初始化权重，利于训练收敛。"""
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        nn.init.xavier_uniform_(self.state_proj.weight)
        nn.init.zeros_(self.state_proj.bias)
        for layer in self.policy_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _encode_history(
            self, entities: torch.Tensor, relations: torch.Tensor, deltas: torch.Tensor
    ) -> torch.Tensor:
        """
        编码历史序列。
        :param entities: 历史实体序列 (Batch, Seq_Len)
        :param relations: 历史关系序列 (Batch, Seq_Len)
        :param deltas: 历史时间差序列 (Batch, Seq_Len)
        :return: 编码后的上下文矩阵 (Batch, Seq_Len, Embed_Dim)
        """
        # 1. 融合特征：实体 + 关系 + 时间
        # 形状: (Batch, Seq_Len, Embed_Dim)
        seq = (
                self.entity_emb(entities)
                + self.relation_emb(relations)
                + self.time_enc(deltas)
        )

        # 2. 加入位置编码
        seq = self.pos_encoder(seq)

        # 3. 维度置换: (Batch, Seq, Dim) -> (Seq, Batch, Dim)
        # 必须这样做，因为 PyTorch 的 TransformerEncoder 默认期待 Seq 在第一维
        seq = seq.transpose(0, 1)

        # 4. Transformer 编码
        context = self.transformer(seq)

        # 5. 换回维度: (Seq, Batch, Dim) -> (Batch, Seq, Dim)
        return context.transpose(0, 1)

    def _compose_state(
            self,
            context_vec: torch.Tensor,
            current_entities: torch.Tensor,
            query_relation: torch.Tensor,
            last_delta: torch.Tensor,
    ) -> torch.Tensor:
        """
        组合当前状态向量 (State Representation)。
        将历史信息与当前节点、目标关系结合，形成“我现在的处境和目标”向量。
        """
        pieces = [
            context_vec,  # 历史摘要 (Batch, Dim)
            self.entity_emb(current_entities),  # 当前所在节点 (Batch, Dim)
            self.relation_emb(query_relation),  # 目标查询关系 (Batch, Dim)
            self.time_enc(last_delta.unsqueeze(1)).squeeze(1),  # 距离上次的时间 (Batch, Dim)
        ]
        # 拼接并投影: [Batch, 4*Dim] -> [Batch, Dim]
        return torch.tanh(self.state_proj(torch.cat(pieces, dim=-1)))

    def _encode_candidates(
            self,
            candidate_entities: torch.Tensor,
            candidate_relations: torch.Tensor,
            candidate_deltas: torch.Tensor,
    ) -> torch.Tensor:
        """
        编码候选邻居 (Next Hops)。
        :return: 候选动作的特征向量 (Batch, Num_Candidates, Dim)
        """
        return (
                self.entity_emb(candidate_entities)
                + self.relation_emb(candidate_relations)
                + self.time_enc(candidate_deltas)
        )

    @torch.jit.export
    def forward(  # type: ignore[override]
            self,
            history_entities: torch.Tensor,
            history_relations: torch.Tensor,
            history_deltas: torch.Tensor,
            current_entities: torch.Tensor,
            query_relation: torch.Tensor,
            candidate_entities: torch.Tensor,
            candidate_relations: torch.Tensor,
            candidate_deltas: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播主函数：计算候选边的得分 (Logits)。
        输入通常是 Batch 形式。

        :param history_*: 历史序列信息 (Batch, Seq_Len)
        :param current_entities: 当前头实体 (Batch)
        :param query_relation: 目标关系 (Batch)
        :param candidate_*: 候选尾实体/关系/时间 (Batch, Num_Candidates)
        :return: 每个候选的得分 logits (Batch, Num_Candidates)
        """

        # 1. 编码历史序列
        context = self._encode_history(history_entities, history_relations, history_deltas)

        # 2. 提取上下文向量 (取序列的最后一个时间步作为 Summarization)
        # context[:, -1, :] 形状: (Batch, Dim)
        context_vec = context[:, -1, :]

        # 3. 组合生成当前的状态查询向量 (State Query Vector)
        # history_deltas[:, -1] 是最近一次交互的时间差
        state = self._compose_state(
            context_vec,
            current_entities,
            query_relation,
            history_deltas[:, -1],
        )

        # 4. 编码所有的候选者 (Key/Value)
        cand_feats = self._encode_candidates(
            candidate_entities, candidate_relations, candidate_deltas
        )

        # 5. 计算匹配得分
        # state.unsqueeze(1) -> (Batch, 1, Dim)
        # cand_feats         -> (Batch, Num_Candidates, Dim)
        # 两者相加会自动广播 (Broadcasting)，模拟 Query 与 Key 的交互
        # 最后通过 MLP (policy_head) 得到分数
        logits = self.policy_head(torch.tanh(state.unsqueeze(1) + cand_feats)).squeeze(-1)

        return logits


__all__ = ["TemporalSARL"]