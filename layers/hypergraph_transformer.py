"""
Hypergraph Transformer Layer (pure PyTorch, no DGL dependency).

Design
------
Treat the hypergraph as a bipartite graph between:
  - node tokens (N)
  - hyperedge tokens (E)

A single transformer block performs attention on concatenated tokens [N + E],
with an attention mask derived from incidence matrix H:
  - Node <-> Hyperedge attention is allowed only when H[n, e] > 0.
  - Node <-> Node and Hyperedge <-> Hyperedge attention are configurable.

Input/Output convention is compatible with HypergraphConv / HypergraphAttention:
  H: [N, E] incidence matrix (non-negative)
  X: [N, in_dim] node features
  out: [N, out_dim]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _HypergraphTransformerBlock(nn.Module):
    """Single transformer block over node+hyperedge tokens."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
        mlp_ratio: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop_path = nn.Dropout(dropout)

        hidden_dim = max(dim, int(dim * mlp_ratio))
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, tokens: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # tokens: [1, N+E, D]
        x = self.norm1(tokens)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        tokens = tokens + self.drop_path(attn_out)

        x2 = self.norm2(tokens)
        mlp_out = self.mlp(x2)
        tokens = tokens + self.drop_path(mlp_out)
        return tokens


class HypergraphTransformer(nn.Module):
    """Hypergraph Transformer with bipartite-structure attention mask.

    Args:
        in_dim: Input node feature dimension.
        out_dim: Output node feature dimension.
        model_dim: Internal transformer token dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer blocks.
        dropout: Dropout rate for attention and MLP.
        mlp_ratio: Expansion ratio for feed-forward hidden dimension.
        allow_node_to_node: If True, allow node<->node attention.
        allow_edge_to_edge: If True, allow hyperedge<->hyperedge attention.
        bias: Whether output projection uses bias.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        mlp_ratio: float = 2.0,
        allow_node_to_node: bool = False,
        allow_edge_to_edge: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if model_dim % num_heads != 0:
            raise ValueError(
                f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.model_dim = model_dim
        self.allow_node_to_node = allow_node_to_node
        self.allow_edge_to_edge = allow_edge_to_edge

        self.node_in = nn.Linear(in_dim, model_dim, bias=False)
        self.edge_in = nn.Linear(in_dim, model_dim, bias=False)

        self.blocks = nn.ModuleList(
            [
                _HypergraphTransformerBlock(
                    dim=model_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(num_layers)
            ]
        )

        self.node_out = nn.Linear(model_dim, out_dim, bias=bias)

    @staticmethod
    def _build_edge_features(H: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        # Weighted average node -> hyperedge features.
        edge_degree = H.sum(dim=0).clamp(min=1e-8)     # [E]
        edge_feat = (H.T / edge_degree.unsqueeze(1)) @ X  # [E, in_dim]
        return edge_feat

    def _build_attn_mask(self, H: torch.Tensor) -> torch.Tensor:
        """Create boolean mask for MultiheadAttention.

        MultiheadAttention mask semantics:
            True  -> position is masked (disallowed)
            False -> position is allowed

        Returns:
            attn_mask: [N+E, N+E], bool
        """
        n_nodes, n_edges = H.shape
        total = n_nodes + n_edges

        # Start from all blocked then open valid routes.
        allow = torch.zeros((total, total), dtype=torch.bool, device=H.device)

        membership = (H > 0)  # [N, E]

        # Node <-> Hyperedge routes from incidence.
        allow[:n_nodes, n_nodes:] = membership
        allow[n_nodes:, :n_nodes] = membership.T

        # Optional dense intra-type routes.
        if self.allow_node_to_node:
            allow[:n_nodes, :n_nodes] = True
        if self.allow_edge_to_edge:
            allow[n_nodes:, n_nodes:] = True

        # Always allow token self-attention.
        eye = torch.eye(total, device=H.device, dtype=torch.bool)
        allow = allow | eye

        # Attention mask requires True for blocked locations.
        attn_mask = ~allow
        return attn_mask

    def forward(self, H: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: incidence matrix [N, E], non-negative.
            X: node features [N, in_dim].
        Returns:
            out: node features [N, out_dim].
        """
        if H.dim() != 2 or X.dim() != 2:
            raise ValueError(f"Expected H and X to be 2D, got H:{tuple(H.shape)} X:{tuple(X.shape)}")
        if H.shape[0] != X.shape[0]:
            raise ValueError(f"H and X must have same node count, got {H.shape[0]} vs {X.shape[0]}")

        n_nodes, _ = H.shape

        edge_feat = self._build_edge_features(H, X)     # [E, in_dim]
        node_tokens = self.node_in(X)                   # [N, D]
        edge_tokens = self.edge_in(edge_feat)           # [E, D]

        tokens = torch.cat([node_tokens, edge_tokens], dim=0).unsqueeze(0)  # [1, N+E, D]
        attn_mask = self._build_attn_mask(H)                                 # [N+E, N+E]

        for block in self.blocks:
            tokens = block(tokens, attn_mask)

        node_tokens_out = tokens[:, :n_nodes, :].squeeze(0)  # [N, D]
        out = self.node_out(node_tokens_out)                 # [N, out_dim]
        return out
