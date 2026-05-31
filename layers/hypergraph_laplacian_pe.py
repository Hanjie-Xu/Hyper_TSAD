import torch
import torch.nn as nn


class HypergraphLaplacianPE(nn.Module):
    """Compute node positional embeddings from hypergraph Laplacian eigenvectors.

    Laplacian:
        L = I - D_v^{-1/2} H B D_e^{-1} H^T D_v^{-1/2}

    with B set to identity by default.
    Returns the smallest k eigenvectors U_k in R^{N x k}.
    """

    def __init__(self, k: int = 8, eps: float = 1e-8):
        super().__init__()
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = int(k)
        self.eps = float(eps)

    def _stable_smallest_eigvecs(self, lap: torch.Tensor) -> torch.Tensor:
        """Return eigenvectors for smallest eigenvalues with robust fallbacks."""
        n = lap.shape[0]
        device = lap.device
        out_dtype = lap.dtype

        # eigh can be fragile on ill-conditioned float32 inputs; solve in float64.
        lap64 = lap.to(dtype=torch.float64)
        eye64 = torch.eye(n, dtype=torch.float64, device=device)

        # Retry with progressively stronger diagonal jitter.
        jitter_values = [0.0, 1e-10, 1e-8, 1e-6, 1e-4]
        for jitter in jitter_values:
            try:
                lap_try = lap64 if jitter == 0.0 else (lap64 + jitter * eye64)
                _, eigvecs = torch.linalg.eigh(lap_try)
                return eigvecs.to(dtype=out_dtype)
            except RuntimeError:
                continue

        # Final fallback: SVD on symmetric Laplacian, then use left singular vectors.
        # For PSD Laplacians this is a practical substitute for eigenvectors.
        u, _, _ = torch.linalg.svd(lap64, full_matrices=False)
        return u.to(dtype=out_dtype)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: incidence matrix [N, E], non-negative
        Returns:
            U_k: [N, k] smallest Laplacian eigenvectors (zero-padded if k > N)
        """
        if H.dim() != 2:
            raise ValueError(f"Expected 2D incidence matrix H, got shape {tuple(H.shape)}")

        H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
        n_nodes, _ = H.shape
        device = H.device
        dtype = H.dtype

        dv = H.sum(dim=1).clamp(min=self.eps)  # [N]
        de = H.sum(dim=0).clamp(min=self.eps)  # [E]

        dv_inv_sqrt = torch.diag(dv.pow(-0.5))
        de_inv = torch.diag(de.pow(-1.0))

        # Hyperedge weight matrix B defaults to identity.
        # H B D_e^{-1} == H D_e^{-1} when B=I.
        a = dv_inv_sqrt @ H @ de_inv @ H.transpose(0, 1) @ dv_inv_sqrt
        eye = torch.eye(n_nodes, dtype=dtype, device=device)
        lap = eye - a

        # Numerical symmetrisation before eigendecomposition.
        lap = 0.5 * (lap + lap.transpose(0, 1))
        eigvecs = self._stable_smallest_eigvecs(lap)

        k_eff = min(self.k, n_nodes)
        uk = eigvecs[:, :k_eff]

        if k_eff < self.k:
            pad = torch.zeros((n_nodes, self.k - k_eff), dtype=dtype, device=device)
            uk = torch.cat([uk, pad], dim=1)

        return uk
