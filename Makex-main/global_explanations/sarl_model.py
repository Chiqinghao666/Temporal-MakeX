#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Temporal-SARL transformer policy network."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    """Continuous-time encoding combining linear and periodic terms."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError("Time2Vec dimension must be >= 2")
        self.linear_weight = nn.Parameter(torch.randn(1))
        self.linear_bias = nn.Parameter(torch.zeros(1))
        self.freq = nn.Parameter(torch.randn(dim - 1))
        self.phase = nn.Parameter(torch.zeros(dim - 1))

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        delta = delta.unsqueeze(-1)
        linear = self.linear_weight * delta + self.linear_bias
        sinusoid = torch.sin(delta * self.freq.view(1, 1, -1) + self.phase.view(1, 1, -1))
        return torch.cat([linear, sinusoid], dim=-1)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

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


class TemporalSARL(nn.Module):
    """Temporal-aware transformer scoring module."""

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
        self.entity_emb = nn.Embedding(num_entities, embed_dim)
        self.relation_emb = nn.Embedding(num_relations, embed_dim)
        self.time_enc = Time2Vec(embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dropout=dropout, batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.state_proj = nn.Linear(embed_dim * 4, embed_dim)
        self.policy_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
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
        seq = (
            self.entity_emb(entities)
            + self.relation_emb(relations)
            + self.time_enc(deltas)
        )
        seq = self.pos_encoder(seq)
        seq = seq.transpose(0, 1)
        context = self.transformer(seq)
        return context.transpose(0, 1)

    def _compose_state(
        self,
        context_vec: torch.Tensor,
        current_entities: torch.Tensor,
        query_relation: torch.Tensor,
        last_delta: torch.Tensor,
    ) -> torch.Tensor:
        pieces = [
            context_vec,
            self.entity_emb(current_entities),
            self.relation_emb(query_relation),
            self.time_enc(last_delta.unsqueeze(1)).squeeze(1),
        ]
        return torch.tanh(self.state_proj(torch.cat(pieces, dim=-1)))

    def _encode_candidates(
        self,
        candidate_entities: torch.Tensor,
        candidate_relations: torch.Tensor,
        candidate_deltas: torch.Tensor,
    ) -> torch.Tensor:
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
        """Return logits for each candidate edge."""

        context = self._encode_history(history_entities, history_relations, history_deltas)
        context_vec = context[:, -1, :]
        state = self._compose_state(
            context_vec,
            current_entities,
            query_relation,
            history_deltas[:, -1],
        )
        cand_feats = self._encode_candidates(
            candidate_entities, candidate_relations, candidate_deltas
        )
        logits = self.policy_head(torch.tanh(state.unsqueeze(1) + cand_feats)).squeeze(-1)
        return logits


__all__ = ["TemporalSARL"]
