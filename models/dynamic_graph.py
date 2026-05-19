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
            H: [N, N] incidence matrix (float, on the same device as node_feat).
               H[n, e] = 1 if node n is a member of hyperedge e, else 0.
        """
        N = node_feat.shape[0]
        k = min(self.topk, N - 1)

        if self.similarity_metric == 'cosine':
            norm = torch.norm(node_feat, p=2, dim=1, keepdim=True).clamp(min=1e-8)
            feat = node_feat / norm
        else:
            feat = node_feat

        sim = torch.matmul(feat, feat.T)  # [N, N]
        sim.fill_diagonal_(-1e9)          # exclude self when finding neighbours

        _, topk_idx = torch.topk(sim, k, dim=1)  # [N, k]

        # Build hard incidence matrix
        H = torch.zeros(N, N, dtype=node_feat.dtype, device=node_feat.device)
        H.fill_diagonal_(1.0)  # every node belongs to its own hyperedge
        # each row e: mark k neighbours as members of hyperedge e
        row_idx = torch.arange(N, device=node_feat.device).unsqueeze(1).expand(N, k)
        H[topk_idx, row_idx] = 1.0  # H[neighbour_n, edge_e] = 1

        return H  # [N, E=N]