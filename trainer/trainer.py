import math

import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        device,
        w_mse=1.0,
        w_graph_diff=0.01,
        w_graph_sparse=0.01,
        score_aggregation='topk',
        score_topk_ratio=0.2,
        score_normalize=False,
        score_horizons=3,
    ):

        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.forecast_loss = nn.MSELoss()
        self.w_mse = float(w_mse)
        self.w_graph_diff = float(w_graph_diff)
        self.w_graph_sparse = float(w_graph_sparse)
        self.score_aggregation = score_aggregation
        self.score_topk_ratio = float(score_topk_ratio)
        self.score_normalize = bool(score_normalize)
        self.score_horizons = max(1, int(score_horizons))
        self.global_step = 0
        self._calib_mean = None
        self._calib_std = None

        if self.score_aggregation not in {'mean', 'topk'}:
            raise ValueError(
                f"score_aggregation must be one of {{'mean', 'topk'}}, got {self.score_aggregation}"
            )
        if not (0.0 < self.score_topk_ratio <= 1.0):
            raise ValueError('score_topk_ratio must be in the interval (0, 1].')

    def _graph_temporal_diff(self, A_cur, A_prev):
        return torch.mean(torch.abs(A_cur - A_prev))

    def _graph_sparsity(self, A_cur):
        return torch.mean(A_cur)

    def _window_anomaly_score(self, pred_delta, true_delta):
        # residual: [B, N] – mean over the (single) time dimension
        residual = torch.abs(pred_delta - true_delta).mean(dim=1)

        # Per-variable Z-score normalisation using calibration statistics.
        # This prevents high-variance variables from dominating the aggregate score
        # and makes thresholding on calibration data more reliable.
        if self.score_normalize and self._calib_mean is not None:
            calib_mean = self._calib_mean.to(residual.device)
            calib_std = self._calib_std.to(residual.device)
            residual = (residual - calib_mean) / calib_std
            residual = residual.clamp(min=0.0)  # keep only above-baseline deviations

        if self.score_aggregation == 'mean':
            return residual.mean(dim=1)

        num_vars = residual.shape[1]
        k = max(1, math.ceil(num_vars * self.score_topk_ratio))
        topk_values, _ = torch.topk(residual, k=k, dim=1)
        return topk_values.mean(dim=1)

    @torch.no_grad()
    def calibrate(self, loader):
        """Compute per-variable residual mean/std on normal training windows.

        Must be called after training, before inference, when score_normalize=True.
        Iterates over all horizons to match the multi-horizon scoring used at test time.
        """
        self.model.eval()
        all_residuals = []
        for batch in loader:
            x = batch['x'].to(self.device)
            T = x.shape[1]
            for h in range(1, self.score_horizons + 1):
                if T - h < 1:
                    break
                x_hist = x[:, :T - h, :]
                y_true = x[:, T - h: T - h + 1, :]
                z_pred, _ = self.model(
                    x_hist,
                    force_graph_rebuild=True,
                    update_graph_cache=False,
                )
                x_last = x_hist[:, -1:, :]
                pred_delta = z_pred - x_last
                true_delta = y_true - x_last
                residual = torch.abs(pred_delta - true_delta).mean(dim=1)  # [B, N]
                all_residuals.append(residual.cpu())

        all_residuals = torch.cat(all_residuals, dim=0)  # [M, N]
        self._calib_mean = all_residuals.mean(dim=0)     # [N]
        self._calib_std = all_residuals.std(dim=0).clamp(min=1e-8)  # [N]
    
    def train_epoch(self, loader):

        self.model.train()
        total_loss = 0
        pbar = tqdm(loader)

        for batch in pbar:
            x = batch['x'].to(self.device)
            # x_hist: [B, T-1, N], y_true: [B, 1, N]
            x_hist = x[:, :-1, :]
            y_true = x[:, -1:, :]

            graph_update_freq = int(getattr(self.model, 'graph_update_freq', 1))
            should_refresh_graph = (self.global_step % max(graph_update_freq, 1) == 0)

            self.optimizer.zero_grad()

            z_pred, A_cur = self.model(
                x_hist,
                force_graph_rebuild=should_refresh_graph,
                update_graph_cache=True
            )
            # Residual prediction: predict delta = y_{t+1} - y_t
            # z_pred: [B, 1, N], x_last: [B, 1, N], y_true: [B, 1, N]
            x_last = x[:, -2:-1, :]
            pred_delta = z_pred - x_last
            true_delta = y_true - x_last
            loss_mse = self.forecast_loss(pred_delta, true_delta)

            if x_hist.shape[1] > 1:
                x_prev = x[:, :-2, :]
                _, A_prev = self.model(
                    x_prev,
                    force_graph_rebuild=should_refresh_graph,
                    update_graph_cache=False
                )
                loss_graph_diff = self._graph_temporal_diff(A_cur, A_prev)
            else:
                loss_graph_diff = torch.zeros(1, device=self.device, dtype=loss_mse.dtype).squeeze(0)

            # Skip sparsity regularisation for hypergraph mode: the soft incidence
            # matrix has a well-defined mean (≈ k/N + 1/N), and penalising it would
            # push membership weights toward zero and degrade aggregation quality.
            is_hypergraph = (
                getattr(self.model, 'graph_ablation', '') == 'dynamic_hypergraph'
            )
            if is_hypergraph:
                loss_graph_sparse = torch.zeros(1, device=self.device, dtype=loss_mse.dtype).squeeze(0)
            else:
                loss_graph_sparse = self._graph_sparsity(A_cur)

            loss = (
                self.w_mse * loss_mse
                + self.w_graph_diff * loss_graph_diff
                + self.w_graph_sparse * loss_graph_sparse
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                5.0
            )
            self.optimizer.step()
            self.global_step += 1
            total_loss += loss.item()
            pbar.set_description(
                f'loss={loss.item():.4f} mse={loss_mse.item():.4f} '
                f'diff={loss_graph_diff.item():.4f} sparse={loss_graph_sparse.item():.4f}'
            )
        return total_loss / len(loader)
    
    @torch.no_grad()
    def inference(self, loader):
        """Run anomaly scoring over a dataloader.

        Multi-horizon scoring: for each window of size T, predictions are
        computed for the last `score_horizons` steps (h=1..score_horizons),
        each using a different-length prefix as history.  Scores are averaged
        across horizons, giving a more robust estimate than single-step scoring.
        """
        self.model.eval()
        all_scores = []
        for batch in loader:
            x = batch['x'].to(self.device)
            T = x.shape[1]
            horizon_scores = []
            for h in range(1, self.score_horizons + 1):
                if T - h < 1:
                    break
                x_hist = x[:, :T - h, :]
                y_true = x[:, T - h: T - h + 1, :]
                z_pred, _ = self.model(
                    x_hist,
                    force_graph_rebuild=True,
                    update_graph_cache=False,
                )
                x_last = x_hist[:, -1:, :]
                pred_delta = z_pred - x_last
                true_delta = y_true - x_last
                score_h = self._window_anomaly_score(pred_delta, true_delta)  # [B]
                horizon_scores.append(score_h)

            # Average across prediction horizons [n_horizons, B] → [B]
            score = torch.stack(horizon_scores, dim=0).mean(dim=0)
            all_scores.append(score.cpu())
        return torch.cat(all_scores)