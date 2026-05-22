"""
Hypergraph Attention Layer (pure PyTorch, no DGL dependency).

This layer follows the same two-stage message passing as HGNN:
  1) Node -> Hyperedge
  2) Hyperedge -> Node

But replaces uniform incidence weighting with learned attention in both stages.

Input/Output convention is compatible with HypergraphConv:
  H: [N, E] incidence matrix (non-negative)
  X: [N, in_dim] node features
  out: [N, out_dim]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HypergraphAttention(nn.Module):
    """Single hypergraph attention layer.

    Args:
        in_dim (int): Input feature dimension.
        out_dim (int): Output feature dimension.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout applied to attention coefficients.
        negative_slope (float): LeakyReLU negative slope in attention logits.
        concat (bool): If True, concatenate multi-head outputs, otherwise average.
        bias (bool): Whether to include bias in output projection.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
        concat: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.concat = concat

        self.head_dim = out_dim if not concat else max(1, out_dim // num_heads)
        proj_out_dim = self.head_dim * num_heads

        # Shared node projection for both stages.
        self.node_proj = nn.Linear(in_dim, proj_out_dim, bias=False)

        # Stage-1 attention (node -> hyperedge): alpha(n,e)
        self.attn_n2e_node = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.attn_n2e_edge = nn.Parameter(torch.empty(num_heads, self.head_dim))

        # Stage-2 attention (hyperedge -> node): beta(e,n)
        self.attn_e2n_edge = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.attn_e2n_node = nn.Parameter(torch.empty(num_heads, self.head_dim))

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.attn_dropout = nn.Dropout(dropout)

        if concat:
            self.out_proj = nn.Linear(proj_out_dim, out_dim, bias=bias)
        else:
            self.out_proj = nn.Linear(self.head_dim, out_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.node_proj.weight)
        nn.init.xavier_uniform_(self.attn_n2e_node)
        nn.init.xavier_uniform_(self.attn_n2e_edge)
        nn.init.xavier_uniform_(self.attn_e2n_edge)
        nn.init.xavier_uniform_(self.attn_e2n_node)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _masked_softmax(self, logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        # logits: [N,E,H] or [E,N,H], mask same leading dims without H or with broadcastable H
        masked_logits = logits.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(masked_logits, dim=dim)
        attn = torch.where(torch.isfinite(attn), attn, torch.zeros_like(attn))
        return self.attn_dropout(attn)

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

        N, E = H.shape
        device = X.device
        eps = 1e-8

        # Binary membership mask from incidence.
        membership = (H > 0).to(device)  # [N, E]

        # Node projection: [N, H, D]
        Xp = self.node_proj(X).view(N, self.num_heads, self.head_dim)

        # Build initial edge features with incidence-weighted average.
        edge_degree = H.sum(dim=0).clamp(min=eps)                  # [E]
        edge_feat = (H.T / edge_degree.unsqueeze(1)) @ X           # [E, in_dim]
        Ep = self.node_proj(edge_feat).view(E, self.num_heads, self.head_dim)  # [E,H,D]

        # ---------- Stage 1: Node -> Hyperedge attention ----------
        # logits_n2e[n,e,h] = a_n[h]^T Xp[n,h] + a_e[h]^T Ep[e,h]
        score_n = (Xp * self.attn_n2e_node.unsqueeze(0)).sum(dim=-1)  # [N,H]
        score_e = (Ep * self.attn_n2e_edge.unsqueeze(0)).sum(dim=-1)   # [E,H]
        logits_n2e = self.leaky_relu(score_n.unsqueeze(1) + score_e.unsqueeze(0))  # [N,E,H]

        # Mask non-membership and softmax over nodes inside each hyperedge.
        mask_n2e = membership.unsqueeze(-1).expand(N, E, self.num_heads)
        alpha = self._masked_softmax(logits_n2e, mask_n2e, dim=0)  # [N,E,H]

        # edge_msg[e,h,d] = sum_n alpha[n,e,h] * Xp[n,h,d]
        edge_msg = torch.einsum("neh,nhd->ehd", alpha, Xp)  # [E,H,D]

        # ---------- Stage 2: Hyperedge -> Node attention ----------
        # logits_e2n[e,n,h] = b_e[h]^T edge_msg[e,h] + b_n[h]^T Xp[n,h]
        score_edge = (edge_msg * self.attn_e2n_edge.unsqueeze(0)).sum(dim=-1)  # [E,H]
        score_node = (Xp * self.attn_e2n_node.unsqueeze(0)).sum(dim=-1)         # [N,H]
        logits_e2n = self.leaky_relu(score_edge.unsqueeze(1) + score_node.unsqueeze(0))  # [E,N,H]

        # Same membership, but now softmax over hyperedges for each node.
        mask_e2n = membership.T.unsqueeze(-1).expand(E, N, self.num_heads)
        beta = self._masked_softmax(logits_e2n, mask_e2n, dim=0)  # [E,N,H]

        # node_msg[n,h,d] = sum_e beta[e,n,h] * edge_msg[e,h,d]
        node_msg = torch.einsum("enh,ehd->nhd", beta, edge_msg)  # [N,H,D]

        if self.concat:
            out = node_msg.reshape(N, self.num_heads * self.head_dim)
        else:
            out = node_msg.mean(dim=1)

        return self.out_proj(out)  # [N, out_dim]
