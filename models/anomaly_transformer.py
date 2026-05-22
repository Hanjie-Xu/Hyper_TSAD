import torch
import torch.nn as nn

from layers.anomaly_transformer_layers import AnomalyAttention, AttentionLayer, DataEmbedding, Encoder, EncoderLayer


class AnomalyTransformerModel(nn.Module):
    """Anomaly Transformer adapted to next-step prediction in this project."""

    def __init__(
        self,
        num_vars: int,
        window_size: int,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.num_vars = int(num_vars)
        self.window_size = int(window_size)

        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, got d_model={d_model}, n_heads={n_heads}")

        self.embedding = DataEmbedding(c_in=self.num_vars, d_model=d_model, dropout=dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(
                            win_size=self.window_size,
                            mask_flag=False,
                            attention_dropout=dropout,
                            output_attention=True,
                        ),
                        d_model=d_model,
                        n_heads=n_heads,
                    ),
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.projection = nn.Linear(d_model, self.num_vars, bias=True)

        # Trainer compatibility flag.
        self.graph_ablation = "none"

    def forward(self, x, force_graph_rebuild=False, update_graph_cache=True):
        del force_graph_rebuild, update_graph_cache

        # x: [B, T, N]
        enc_out = self.embedding(x)
        enc_out, _, _, _ = self.encoder(enc_out)
        out = self.projection(enc_out)  # [B, T, N]
        pred = out[:, -1:, :]           # next-step proxy from the latest token

        bsz, _, n_vars = pred.shape
        adjacency = torch.zeros((bsz, n_vars, n_vars), dtype=pred.dtype, device=pred.device)
        return pred, adjacency
