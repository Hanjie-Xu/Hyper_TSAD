import math

import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer

from layers.tranad_layers import PositionalEncoding


class TranADModel(nn.Module):
    """TranAD-style two-phase transformer adapted to this project's trainer API."""

    def __init__(
        self,
        num_vars: int,
        window_size: int,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_vars = int(num_vars)
        self.window_size = int(window_size)
        d_model = 2 * self.num_vars

        # Keep heads valid for any feature size.
        n_heads = max(1, min(self.num_vars, 8))
        while d_model % n_heads != 0 and n_heads > 1:
            n_heads -= 1

        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max(window_size, 512))
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=1)

        decoder_layer1 = TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=False,
        )
        decoder_layer2 = TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer_decoder1 = TransformerDecoder(decoder_layer1, num_layers=1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layer2, num_layers=1)

        self.fcn = nn.Sequential(nn.Linear(d_model, self.num_vars), nn.Sigmoid())

        # Trainer compatibility flag.
        self.graph_ablation = "none"

    def _encode(self, src: torch.Tensor, cond: torch.Tensor, tgt: torch.Tensor):
        # src/cond: [T, B, N], tgt: [1, B, N]
        src_cat = torch.cat((src, cond), dim=-1) * math.sqrt(self.num_vars)
        src_cat = self.pos_encoder(src_cat)
        memory = self.transformer_encoder(src_cat)
        tgt_cat = tgt.repeat(1, 1, 2)
        return tgt_cat, memory

    def forward(self, x, force_graph_rebuild=False, update_graph_cache=True):
        del force_graph_rebuild, update_graph_cache

        # x: [B, T, N]
        src = x.permute(1, 0, 2)  # [T, B, N]
        tgt = src[-1:, :, :]      # [1, B, N]

        cond1 = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self._encode(src, cond1, tgt)))

        # Use first-pass residual as anomaly-aware condition for phase 2.
        cond2 = (x1 - tgt).pow(2).repeat(src.shape[0], 1, 1)
        x2 = self.fcn(self.transformer_decoder2(*self._encode(src, cond2, tgt)))

        pred = x2.permute(1, 0, 2)  # [B, 1, N]
        bsz, _, n_vars = pred.shape
        adjacency = torch.zeros((bsz, n_vars, n_vars), dtype=pred.dtype, device=pred.device)
        return pred, adjacency
