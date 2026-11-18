"""Small LSTM and Transformer-encoder regressors of log volatility (report section 2)."""
from __future__ import annotations

import math

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    """LSTM(32) -> Dropout -> Dense(16, ReLU) -> Dense(1, linear)."""

    def __init__(self, n_features: int, hidden: int = 32, dense: int = 16, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, dense), nn.ReLU(), nn.Linear(dense, 1))

    def forward(self, x):                      # x: (B, T, F)
        _, (h, _) = self.lstm(x)
        return self.head(self.drop(h[-1])).squeeze(-1)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal encoding (Vaswani et al., 2017); own implementation."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerRegressor(nn.Module):
    """Linear projection(32) + positional encoding -> 1 encoder block (2 heads, FF 64, residual+LayerNorm+dropout)
    -> global average pooling over the 60 days -> Dropout -> Dense(16, ReLU) -> Dense(1).

    No causal mask: every row of the window is <= t, so attending to all of them is not future leakage.
    """

    def __init__(self, n_features: int, d_model: int = 32, heads: int = 2, ff: int = 64, blocks: int = 1,
                 dense: int = 16, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=blocks, enable_nested_tensor=False)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(d_model, dense), nn.ReLU(), nn.Linear(dense, 1))

    def forward(self, x):
        h = self.encoder(self.pos(self.proj(x)))
        return self.head(self.drop(h.mean(dim=1))).squeeze(-1)


def build_model(arch: str, n_features: int, cfg: dict, dropout: float) -> nn.Module:
    m = cfg["models"]
    if arch == "LSTM":
        return LSTMRegressor(n_features, m["lstm"]["hidden"], m["lstm"]["dense"], dropout)
    if arch == "Transformer":
        t = m["transformer"]
        return TransformerRegressor(n_features, t["d_model"], t["heads"], t["ff"], t["blocks"], t["dense"], dropout)
    raise ValueError(arch)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
