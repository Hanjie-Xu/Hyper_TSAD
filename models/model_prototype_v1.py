import torch
import torch.nn as nn

import dgl
import dgl.nn as dglnn

from models.dynamic_graph import DynamicGraphBuilder, DynamicHypergraphBuilder
from layers.hypergraph_conv import HypergraphConv


class VariableEncoder(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()

        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim,
            batch_first=True
        )

    def forward(self, x):
        """
        x: [B,T]
        """
        x = x.unsqueeze(-1)
        h, _ = self.gru(x)
        return h

class TemporalEncoder(nn.Module):

    def __init__(self, num_vars, hidden_dim):
        super().__init__()
        self.encoders = nn.ModuleList([
            VariableEncoder(hidden_dim)
            for _ in range(num_vars)
        ])

    def forward(self, x):
        """
        x: [B,T,N]
        return:
            H: [B,T,N,D]
        """
        outputs = []
        for i, encoder in enumerate(self.encoders):
            h = encoder(x[:, :, i])
            outputs.append(h.unsqueeze(2))
        H = torch.cat(outputs, dim=2)
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
        N, H, D = h.shape
        h = h.reshape(N, H * D)
        h = self.proj(h)
        return h

class HypergraphEncoder(nn.Module):
    """Two-layer HGNN encoder with residual connection."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.conv = HypergraphConv(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()

    def forward(self, H: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        H : incidence matrix [N, E]
        x : node features    [N, hidden_dim]
        returns              [N, hidden_dim]
        """
        out = self.conv(H, x)
        out = self.norm(out)
        out = self.act(out)
        return out + x  # residual


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

class ModelPrototype(nn.Module):

    def __init__(
        self,
        num_vars,
        hidden_dim=64,
        proj_dim=32,
        topk=5,
        graph_ablation='dynamic',
        graph_update_freq=1,
        static_graph=None,
        graph_similarity_metric='dot_product'
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

        self.num_vars = num_vars
        self.graph_ablation = graph_ablation
        self.graph_update_freq = graph_update_freq
        self.static_graph = static_graph
        self._cached_dynamic_graphs = None

        self.temporal_encoder = TemporalEncoder(
            num_vars=num_vars,
            hidden_dim=hidden_dim
        )
        self.graph_builder = DynamicGraphBuilder(
            topk=topk,
            similarity_metric=graph_similarity_metric
        )
        self.graph_encoder = GraphEncoder(
            hidden_dim=hidden_dim
        )
        # Hypergraph components (only used when graph_ablation='dynamic_hypergraph')
        self.hypergraph_builder = DynamicHypergraphBuilder(
            topk=topk,
            similarity_metric=graph_similarity_metric
        )
        self.hypergraph_encoder = HypergraphEncoder(hidden_dim=hidden_dim)
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
        """Build hypergraph incidence matrix H [N, N] (E == N)."""
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
                A_b = H_inc  # use incidence matrix as the "adjacency" for loss terms

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
