"""
Hypergraph Convolution Layer (HGNN-style, pure PyTorch, no DGL dependency).

Two-step aggregation:
  1. Node → Hyperedge:  edge_feat = B^{-1} H^T X
  2. Hyperedge → Node:  X'        = D^{-1} H edge_feat

Full formula: X' = D^{-1} H B^{-1} H^T X W
where
  H  : incidence matrix  [N, E]  (H[n,e] > 0 if node n belongs to hyperedge e)
  D  : diagonal node degree   D[n] = sum_e H[n,e]
  B  : diagonal edge degree   B[e] = sum_n H[n,e]
  W  : learnable weight  [in_dim, out_dim]
"""

import torch
import torch.nn as nn


class HypergraphConv(nn.Module):
    """Single HGNN-style hypergraph convolution layer.

    Args:
        in_dim  (int): Input feature dimension.
        out_dim (int): Output feature dimension.
        bias    (bool): Whether to add a bias term after the linear transform.
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, H: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H : incidence matrix  [N, E], non-negative
            X : node features     [N, in_dim]
        Returns:
            out: updated node features  [N, out_dim]
        """
        # --- degree vectors (clamp to avoid div-by-zero) ---
        D_v = H.sum(dim=1).clamp(min=1e-8)  # [N]  node degree
        B_e = H.sum(dim=0).clamp(min=1e-8)  # [E]  edge degree

        # --- Step 1: node → hyperedge aggregation ---
        # edge_agg[e] = sum_n  H[n,e] / B[e]  * X[n]
        # matrix form: (H / B[e]).T @ X  =  (H.T / B[:,None]) @ X
        edge_agg = (H.T / B_e.unsqueeze(1)) @ X        # [E, in_dim]

        # --- Step 2: hyperedge → node aggregation ---
        # node_agg[n] = sum_e  H[n,e] / D[n]  * edge_agg[e]
        # matrix form: (H / D[n,None]) @ edge_agg
        node_agg = (H / D_v.unsqueeze(1)) @ edge_agg   # [N, in_dim]

        return self.linear(node_agg)                    # [N, out_dim]
