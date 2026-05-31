import torch
import torch.nn as nn

import dgl

from models.dynamic_graph import DynamicGraphBuilder, DynamicHypergraphBuilder
from models.model_prototype_v1_5 import GraphEncoder, PredictionHead, TemporalEncoderCausalConv
from layers.hypergraph_laplacian_pe import HypergraphLaplacianPE
from layers.hypergraph_transformer import HypergraphTransformer


class HypergraphEncoderTransformerNoOuterResidual(nn.Module):
    """Hypergraph transformer encoder without outer residual connection."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        mlp_ratio: float = 2.0,
        allow_node_to_node: bool = False,
        allow_edge_to_edge: bool = False,
    ):
        super().__init__()
        self.encoder = HypergraphTransformer(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            model_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            allow_node_to_node=allow_node_to_node,
            allow_edge_to_edge=allow_edge_to_edge,
            bias=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()

    def forward(self, H: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(H, x)
        out = self.norm(out)
        out = self.act(out)
        return out


class ModelPrototypeV15T3(nn.Module):
    """Model v1.5t.3: v1.5t1 + hypergraph Laplacian positional embedding (HLPE)."""

    def __init__(
        self,
        num_vars,
        hidden_dim=64,
        proj_dim=32,
        topk=5,
        graph_ablation='dynamic_hypergraph',
        graph_update_freq=1,
        static_graph=None,
        graph_similarity_metric='dot_product',
        temporal_kernel_size=3,
        temporal_num_layers=3,
        temporal_dropout=0.1,
        temporal_dilation_base=2,
        hypergraph_transformer_heads=4,
        hypergraph_transformer_layers=2,
        hypergraph_transformer_dropout=0.1,
        hypergraph_transformer_mlp_ratio=2.0,
        hypergraph_transformer_allow_node_to_node=False,
        hypergraph_transformer_allow_edge_to_edge=False,
        hlpe_k=8,
        hlpe_scale=1.0,
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
        self.hlpe_scale = float(hlpe_scale)
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
        self.hypergraph_encoder = HypergraphEncoderTransformerNoOuterResidual(
            hidden_dim=hidden_dim,
            num_heads=hypergraph_transformer_heads,
            num_layers=hypergraph_transformer_layers,
            dropout=hypergraph_transformer_dropout,
            mlp_ratio=hypergraph_transformer_mlp_ratio,
            allow_node_to_node=hypergraph_transformer_allow_node_to_node,
            allow_edge_to_edge=hypergraph_transformer_allow_edge_to_edge,
        )

        self.hlpe = HypergraphLaplacianPE(k=hlpe_k)
        self.hlpe_proj = nn.Linear(hlpe_k, hidden_dim, bias=False)

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

                # Hypergraph Laplacian positional embedding (HLPE)
                # U_k in R^{N x k}, then projected to hidden_dim and added to node features.
                pe = self.hlpe(H_inc)
                node_feat_with_pe = node_feat + self.hlpe_scale * self.hlpe_proj(pe)

                h_graph = self.hypergraph_encoder(H_inc, node_feat_with_pe)
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
ModelPrototype = ModelPrototypeV15T3
