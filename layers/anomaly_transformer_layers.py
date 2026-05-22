import math
from math import sqrt
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TriangularCausalMask:
    def __init__(self, batch_size: int, length: int, device: torch.device):
        mask_shape = [batch_size, 1, length, length]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool, device=device), diagonal=1)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask


class AnomalyAttention(nn.Module):
    """Anomaly-Attention from the Anomaly Transformer implementation."""

    def __init__(
        self,
        win_size: int,
        mask_flag: bool = False,
        scale: Optional[float] = None,
        attention_dropout: float = 0.0,
        output_attention: bool = True,
    ):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        distances = torch.zeros((win_size, win_size), dtype=torch.float32)
        for i in range(win_size):
            for j in range(win_size):
                distances[i, j] = abs(i - j)
        self.register_buffer("distances", distances)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        sigma: torch.Tensor,
        attn_mask: Optional[TriangularCausalMask],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        bsz, length, num_heads, head_dim = queries.shape
        _, src_len, _, value_dim = values.shape
        scale = self.scale or 1.0 / sqrt(head_dim)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(bsz, length, device=queries.device)
            scores = scores.masked_fill(attn_mask.mask, -np.inf)

        attn = scale * scores

        sigma = sigma.transpose(1, 2)  # [B, H, L]
        window_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5.0) + 1e-5
        sigma = torch.pow(3.0, sigma) - 1.0
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # [B, H, L, L]

        prior = self.distances[:window_size, :window_size].to(queries.device)
        prior = prior.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1)
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior.pow(2) / (2 * sigma.pow(2)))

        series = self.dropout(torch.softmax(attn, dim=-1))
        v_out = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            return v_out.contiguous(), series, prior, sigma
        return v_out.contiguous(), None, None, None


class AttentionLayer(nn.Module):
    def __init__(
        self,
        attention: AnomalyAttention,
        d_model: int,
        n_heads: int,
        d_keys: Optional[int] = None,
        d_values: Optional[int] = None,
    ):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model, n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, attn_mask: Optional[TriangularCausalMask]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        bsz, length, _ = queries.shape
        _, src_len, _ = keys.shape
        num_heads = self.n_heads

        x = queries
        queries = self.query_projection(queries).view(bsz, length, num_heads, -1)
        keys = self.key_projection(keys).view(bsz, src_len, num_heads, -1)
        values = self.value_projection(values).view(bsz, src_len, num_heads, -1)
        sigma = self.sigma_projection(x).view(bsz, length, num_heads)

        out, series, prior, sigma = self.inner_attention(queries, keys, values, sigma, attn_mask)
        out = out.view(bsz, length, -1)
        return self.out_projection(out), series, prior, sigma


class EncoderLayer(nn.Module):
    def __init__(self, attention: AttentionLayer, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor, attn_mask: Optional[TriangularCausalMask] = None):
        new_x, attn, prior, sigma = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn, prior, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers: List[EncoderLayer], norm_layer: Optional[nn.Module] = None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x: torch.Tensor, attn_mask: Optional[TriangularCausalMask] = None):
        series_list = []
        prior_list = []
        sigma_list = []

        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)
        return x, series_list, prior_list, sigma_list


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        padding = 1
        self.token_conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.token_conv(x.permute(0, 2, 1)).transpose(1, 2)


class DataEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
