#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Temporal SARL model definition.

该模块实现了一个轻量级、时态感知的 Transformer 编码器，用于对候选边进行打分。
模型输入显式包含 Query Relation 的嵌入，并且支持 TorchScript 导出。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSARL(nn.Module):
    """Simple temporal-aware Transformer for SARL path scoring."""

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
        self.embed_dim = embed_dim
        self.entity_emb = nn.Embedding(num_entities, embed_dim)
        self.relation_emb = nn.Embedding(num_relations, embed_dim)
        self.time_proj = nn.Linear(1, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        nn.init.xavier_uniform_(self.time_proj.weight)
        if self.time_proj.bias is not None:
            nn.init.zeros_(self.time_proj.bias)
        for m in self.scorer:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_state(
        self,
        current_entity: torch.Tensor,
        query_relation: torch.Tensor,
        time_diff: torch.Tensor,
    ) -> torch.Tensor:
        """Encode the current state."""
        ent_vec = self.entity_emb(current_entity)
        rel_vec = self.relation_emb(query_relation)
        time_vec = self.time_proj(time_diff.unsqueeze(-1))
        return ent_vec + rel_vec + time_vec

    @torch.jit.export
    def forward(  # type: ignore[override]
        self,
        history_entities: torch.Tensor,
        history_relations: torch.Tensor,
        history_times: torch.Tensor,
        current_entities: torch.Tensor,
        query_relation: torch.Tensor,
        candidate_entities: torch.Tensor,
        candidate_relations: torch.Tensor,
        candidate_time_diff: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scores for candidate edges.

        Args:
            history_entities: (B, L) entity ids along the path.
            history_times: (B, L) time differences (float).
            query_relation: (B,) relation ids for query.
            candidate_entities: (B, K) next-hop entity ids.
            candidate_relations: (B, K) relation ids for edges.
            candidate_time_diff: (B, K) time diffs relative to current step.
        Returns:
            scores: (B, K) logits for each candidate.
        """
        batch_size, hist_len = history_entities.shape
        history_feats = (
            self.entity_emb(history_entities)
            + self.relation_emb(history_relations)
            + self.time_proj(history_times.unsqueeze(-1))
        )
        history_feats = self.pos_encoder(history_feats)
        history_feats = history_feats.transpose(0, 1)  # Fix for Torch 1.8
        context = self.transformer(history_feats)
        context = context.transpose(0, 1)  # Fix for Torch 1.8
        state = context[:, -1, :]
        state = state + self.entity_emb(current_entities) + self.relation_emb(query_relation)

        cand_entity_vec = self.entity_emb(candidate_entities)
        cand_rel_vec = self.relation_emb(candidate_relations)
        cand_time_vec = self.time_proj(candidate_time_diff.unsqueeze(-1))
        cand_feats = cand_entity_vec + cand_rel_vec + cand_time_vec

        scores = self.scorer(
            torch.tanh(state.unsqueeze(1) + cand_feats)
        ).squeeze(-1)
        return scores


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding helping transformer capture order."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


__all__ = ["TemporalSARL"]
