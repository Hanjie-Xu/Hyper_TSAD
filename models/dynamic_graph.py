import torch
import dgl


class DynamicGraphBuilder:

    def __init__(self, topk=5, similarity_metric='dot_product'):
        self.topk = topk
        if similarity_metric not in {'dot_product', 'cosine'}:
            raise ValueError(
                f"similarity_metric must be 'dot_product' or 'cosine', got {similarity_metric}"
            )
        self.similarity_metric = similarity_metric

    def build_graph(self, node_feat):
        """
        node_feat: [N,D]
        Builds a kNN graph using specified similarity metric.
        """

        N = node_feat.shape[0]
        k = min(self.topk, max(N - 1, 1))

        if self.similarity_metric == 'cosine':
            # Normalize node features to unit length
            norm = torch.norm(node_feat, p=2, dim=1, keepdim=True)
            norm = torch.clamp(norm, min=1e-8)
            node_feat_normalized = node_feat / norm
            sim = torch.matmul(node_feat_normalized, node_feat_normalized.T)
        else:
            # dot_product (original)
            sim = torch.matmul(node_feat, node_feat.T)

        sim.fill_diagonal_(-1e9)
        _, indices = torch.topk(sim, k, dim=-1)
        src = []
        dst = []
        for i in range(N):
            neighbors = indices[i]
            for j in neighbors:
                src.append(i)
                dst.append(j.item())
        g = dgl.graph((src, dst), num_nodes=N)
        return g


class DynamicHypergraphBuilder:
    """Builds a soft incidence matrix H ∈ R^{N×N} for HGNN-style convolution.

    Each node acts as a hyperedge centre; its k-NN (by cosine similarity) plus
    itself form the membership of that hyperedge.  The incidence weight is set
    to 1 for members and 0 otherwise (hard assignment, differentiable-free but
    simple and stable).

    Args:
        topk (int): Number of neighbours per hyperedge (excluding the centre).
        similarity_metric (str): 'cosine' or 'dot_product'.
    """

    def __init__(self, topk: int = 5, similarity_metric: str = 'cosine'):
        self.topk = topk
        if similarity_metric not in {'dot_product', 'cosine'}:
            raise ValueError(
                f"similarity_metric must be 'dot_product' or 'cosine', got {similarity_metric}"
            )
        self.similarity_metric = similarity_metric

    def build_incidence(self, node_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_feat: [N, D] node feature matrix.
        Returns:
            H: [N, N] soft incidence matrix (float, on the same device as node_feat).
               H[n, e] is a softmax-normalised similarity weight; non-members are 0.
               Self-hyperedge diagonal is kept at 1.0.

        Soft incidence (vs. hard 0/1):
          - Incidence weights are proportional to cosine/dot-product similarity,
            normalised via softmax over the k nearest neighbours per hyperedge.
          - This makes the graph construction path differentiable w.r.t. feature
            magnitudes (though not w.r.t. neighbour identity due to topk).
          - Provides a more expressive aggregation than uniform hard weights.
        """
        N = node_feat.shape[0]
        k = min(self.topk, N - 1)

        if self.similarity_metric == 'cosine':
            norm = torch.norm(node_feat, p=2, dim=1, keepdim=True).clamp(min=1e-8)
            feat = node_feat / norm
        else:
            feat = node_feat

        sim = torch.matmul(feat, feat.T)  # [N, N] – full similarity matrix

        # Use a masked copy for topk selection (exclude self)
        sim_masked = sim.clone()
        sim_masked.fill_diagonal_(-1e9)
        _, topk_idx = torch.topk(sim_masked, k, dim=1)  # [N, k]

        # --- Build soft incidence via softmax over neighbour similarities ---
        # row_idx[e, j] = e  →  used to index column (hyperedge axis)
        row_idx = torch.arange(N, device=node_feat.device).unsqueeze(1).expand(N, k)

        # Gather sim[neighbour_n, edge_e] for each (edge, member) pair
        member_sims = sim[topk_idx, row_idx]   # [N, k]
        soft_weights = torch.softmax(member_sims, dim=1)  # [N, k], sum=1 per edge

        H = torch.zeros(N, N, dtype=node_feat.dtype, device=node_feat.device)
        H.fill_diagonal_(1.0)           # self-loop: node always in its own hyperedge
        H[topk_idx, row_idx] = soft_weights  # H[member_n, edge_e] = soft weight

        return H  # [N, E=N]