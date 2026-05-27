import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn as dglnn

from models.dynamic_graph import DynamicGraphBuilder, DynamicHypergraphBuilder
from layers.hypergraph_conv import HypergraphConv
from layers.hypergraph_attn import HypergraphAttention


class CausalConv1d(nn.Module):
    """1D causal convolution with left padding only.

    Input shape:  [B, C_in, T]
    Output shape: [B, C_out, T]
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        if kernel_size < 2:
            raise ValueError(f"kernel_size must be >= 2, got {kernel_size}")
        if dilation < 1:
            raise ValueError(f"dilation must be >= 1, got {dilation}")

        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class CausalConvBlock(nn.Module):

    def __init__(self, hidden_dim: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(hidden_dim, hidden_dim, kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(hidden_dim, hidden_dim, kernel_size, dilation=dilation)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)
        return out + residual


class TemporalEncoderCausalConv(nn.Module):
    """Temporal encoder using causal convolutions (replaces per-variable GRU).

    Args:
        num_vars: number of variables.
        hidden_dim: hidden feature dimension per variable.
        kernel_size: causal conv kernel size.
        num_layers: number of residual causal conv blocks.
        dropout: dropout in temporal blocks.
        dilation_base: geometric dilation base, e.g. 2 -> 1,2,4,...
    """

    def __init__(
        self,
        num_vars: int,
        hidden_dim: int,
        kernel_size: int = 3,
        num_layers: int = 3,
        dropout: float = 0.1,
        dilation_base: int = 2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if dilation_base < 1:
            raise ValueError(f"dilation_base must be >= 1, got {dilation_base}")

        self.num_vars = int(num_vars)
        self.hidden_dim = int(hidden_dim)

        self.input_proj = CausalConv1d(1, hidden_dim, kernel_size=kernel_size, dilation=1)

        blocks = []
        for i in range(num_layers):
            dilation = dilation_base ** i
            blocks.append(
                CausalConvBlock(
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, N]
        return: H [B, T, N, D]
        """
        B, T, N = x.shape
        if N != self.num_vars:
            raise ValueError(f"Expected num_vars={self.num_vars}, got input with N={N}")

        # Treat each variable as an independent 1D sequence in a shared temporal encoder.
        x_1d = x.permute(0, 2, 1).contiguous().view(B * N, 1, T)  # [B*N,1,T]
        h = self.input_proj(x_1d)  # [B*N,D,T]
        h = self.blocks(h)         # [B*N,D,T]

        H = h.view(B, N, self.hidden_dim, T).permute(0, 3, 1, 2).contiguous()  # [B,T,N,D]
        return H


class GraphEncoder(nn.Module):

    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.gat1 = dglnn.GATConv(
            in_feats=hidden_dim,
            out_feats=hidden_dim,
            num_heads=num_heads,
            allow_zero_in_degree=True
        )
        self.proj = nn.Linear(
            hidden_dim * num_heads,
            hidden_dim
        )

    def forward(self, g, x):
        """
        x: [N,D]
        """
        h = self.gat1(g, x)
        n_nodes, n_heads, dim = h.shape
        h = h.reshape(n_nodes, n_heads * dim)
        h = self.proj(h)
        return h


class HypergraphEncoder(nn.Module):
    """Hypergraph encoder with selectable backend and residual connection."""

    def __init__(
        self,
        hidden_dim: int,
        encoder_type: str = 'conv',
        attn_heads: int = 4,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        if encoder_type not in {'conv', 'attn'}:
            raise ValueError(f"encoder_type must be one of {{'conv', 'attn'}}, got {encoder_type}")
        self.encoder_type = encoder_type
        if encoder_type == 'attn':
            self.conv = HypergraphAttention(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=attn_heads,
                dropout=attn_dropout,
                concat=False,
            )
        else:
            self.conv = HypergraphConv(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()

    def forward(self, H: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(H, x)
        out = self.norm(out)
        out = self.act(out)
        return out + x


class PredictionHead(nn.Module):

    def __init__(self, hidden_dim, pred_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_dim)
        )

    def forward(self, x):
        return self.net(x)


class ModelPrototypeV15(nn.Module):

    def __init__(
        self,
        num_vars,
        hidden_dim=64,
        proj_dim=32,
        topk=5,
        graph_ablation='dynamic',
        graph_update_freq=1,
        static_graph=None,
        graph_similarity_metric='dot_product',
        hypergraph_encoder_type='conv',
        hypergraph_attn_heads=4,
        hypergraph_attn_dropout=0.1,
        temporal_kernel_size=3,
        temporal_num_layers=3,
        temporal_dropout=0.1,
        temporal_dilation_base=2,
    ):
        super().__init__()

        valid_graph_modes = {
            'dynamic',
            'dynamic_hypergraph',
            'pearson_static',
            'identity',
            'fully_connected',
            'none'
        }
        if graph_ablation not in valid_graph_modes:
            raise ValueError(
                f'graph_ablation must be one of {valid_graph_modes}, got {graph_ablation}'
            )
        if graph_update_freq < 1:
            raise ValueError(
                f'graph_update_freq must be >= 1, got {graph_update_freq}'
            )
        if hypergraph_encoder_type not in {'conv', 'attn'}:
            raise ValueError(
                f"hypergraph_encoder_type must be one of {{'conv', 'attn'}}, got {hypergraph_encoder_type}"
            )

        self.num_vars = num_vars
        self.graph_ablation = graph_ablation
        self.graph_update_freq = graph_update_freq
        self.static_graph = static_graph
        self._cached_dynamic_graphs = None

        self.temporal_encoder = TemporalEncoderCausalConv(
            num_vars=num_vars,
            hidden_dim=hidden_dim,
            kernel_size=temporal_kernel_size,
            num_layers=temporal_num_layers,
            dropout=temporal_dropout,
            dilation_base=temporal_dilation_base,
        )
        self.graph_builder = DynamicGraphBuilder(
            topk=topk,
            similarity_metric=graph_similarity_metric
        )
        self.graph_encoder = GraphEncoder(
            hidden_dim=hidden_dim
        )
        self.hypergraph_builder = DynamicHypergraphBuilder(
            topk=topk,
            similarity_metric=graph_similarity_metric
        )
        self.hypergraph_encoder = HypergraphEncoder(
            hidden_dim=hidden_dim,
            encoder_type=hypergraph_encoder_type,
            attn_heads=hypergraph_attn_heads,
            attn_dropout=hypergraph_attn_dropout,
        )
        self.prediction_head = PredictionHead(
            hidden_dim=hidden_dim,
            pred_dim=1
        )

    def _graph_to_adjacency(self, g, num_nodes, device):
        A = torch.zeros((num_nodes, num_nodes), device=device)
        if g is None:
            return A
        src, dst = g.edges()
        if len(src) > 0:
            A[src.long(), dst.long()] = 1.0
        return A

    def _build_identity_graph(self, num_nodes, device):
        nodes = torch.arange(num_nodes, device=device)
        return dgl.graph((nodes, nodes), num_nodes=num_nodes, device=device)

    def _build_fully_connected_graph(self, num_nodes, device):
        src = []
        dst = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    src.append(i)
                    dst.append(j)
        return dgl.graph((src, dst), num_nodes=num_nodes, device=device)

    def _build_graph(self, node_feat, device):
        num_nodes = node_feat.shape[0]
        if self.graph_ablation == 'dynamic':
            return self.graph_builder.build_graph(node_feat).to(device)
        if self.graph_ablation == 'pearson_static':
            if self.static_graph is None:
                raise ValueError('static_graph must be provided when graph_ablation=pearson_static')
            return self.static_graph.to(device)
        if self.graph_ablation == 'identity':
            return self._build_identity_graph(num_nodes, device)
        if self.graph_ablation == 'fully_connected':
            return self._build_fully_connected_graph(num_nodes, device)
        return None

    def _build_incidence(self, node_feat):
        return self.hypergraph_builder.build_incidence(node_feat)

    def forward(self, x, force_graph_rebuild=False, update_graph_cache=True):
        B, T, N = x.shape
        H = self.temporal_encoder(x)

        is_hypergraph_mode = (self.graph_ablation == 'dynamic_hypergraph')
        is_dynamic_mode = (self.graph_ablation == 'dynamic')
        use_cache = (
            (is_dynamic_mode or is_hypergraph_mode)
            and self.training
            and not force_graph_rebuild
            and self._cached_dynamic_graphs is not None
            and len(self._cached_dynamic_graphs) == B
        )

        pred_outputs = []
        adj_outputs = []
        rebuilt_cache = []
        for b in range(B):
            node_feat = H[b, -1]

            if self.graph_ablation == 'none':
                h_graph = node_feat
                A_b = torch.zeros(N, N, device=x.device)

            elif is_hypergraph_mode:
                if use_cache:
                    H_inc = self._cached_dynamic_graphs[b]
                else:
                    H_inc = self._build_incidence(node_feat)
                    if self.training:
                        rebuilt_cache.append(H_inc)
                h_graph = self.hypergraph_encoder(H_inc, node_feat)
                A_b = H_inc

            else:
                if use_cache:
                    g = self._cached_dynamic_graphs[b]
                else:
                    g = self._build_graph(node_feat, x.device)
                    if is_dynamic_mode and self.training:
                        rebuilt_cache.append(g)
                if g is not None:
                    h_graph = self.graph_encoder(g, node_feat)
                else:
                    h_graph = node_feat
                A_b = self._graph_to_adjacency(g, N, x.device)

            z_b = self.prediction_head(h_graph).squeeze(-1)
            pred_outputs.append(z_b)
            adj_outputs.append(A_b)

        if (
            (is_dynamic_mode or is_hypergraph_mode)
            and self.training
            and update_graph_cache
            and not use_cache
            and len(rebuilt_cache) == B
        ):
            self._cached_dynamic_graphs = rebuilt_cache

        z = torch.stack(pred_outputs, dim=0).unsqueeze(1)
        A = torch.stack(adj_outputs, dim=0)
        return z, A


# Optional compatibility alias for external imports.
ModelPrototype = ModelPrototypeV15
