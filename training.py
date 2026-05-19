import os
import random
from typing import Dict, List, Optional, Tuple

import dgl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models.model_prototype_v1 import ModelPrototype
from trainer.trainer import Trainer


class WindowDataset(Dataset):
    def __init__(self, windows: np.ndarray):
        self.windows = torch.from_numpy(windows).float()

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"x": self.windows[idx]}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe_join(*parts: str) -> str:
    return os.path.normpath(os.path.join(*parts))


def list_available_entities(dataset_dir: str) -> List[str]:
    files = os.listdir(dataset_dir)
    entities = []
    for name in files:
        if name.endswith("_train.npy"):
            entities.append(name[: -len("_train.npy")])
    return sorted(entities)


def resolve_entity_triplet(dataset_dir: str, entity: Optional[str]) -> Tuple[str, str, str]:
    if entity is None:
        train_path = _safe_join(dataset_dir, "train.npy")
        test_path = _safe_join(dataset_dir, "test.npy")
        labels_path = _safe_join(dataset_dir, "labels.npy")
        if all(os.path.exists(p) for p in [train_path, test_path, labels_path]):
            return train_path, test_path, labels_path

        entities = list_available_entities(dataset_dir)
        if not entities:
            raise FileNotFoundError(
                f"Could not find train/test/labels files in {dataset_dir}. "
                "Expected either train.npy/test.npy/labels.npy or <entity>_train.npy triplets."
            )
        raise ValueError(
            "This dataset contains multiple entities. "
            f"Please pass --entity. Available entities: {entities[:10]}"
            + (" ..." if len(entities) > 10 else "")
        )

    train_path = _safe_join(dataset_dir, f"{entity}_train.npy")
    test_path = _safe_join(dataset_dir, f"{entity}_test.npy")
    labels_path = _safe_join(dataset_dir, f"{entity}_labels.npy")
    if not all(os.path.exists(p) for p in [train_path, test_path, labels_path]):
        raise FileNotFoundError(
            "Could not resolve files for requested entity. "
            f"Expected {train_path}, {test_path}, {labels_path}"
        )
    return train_path, test_path, labels_path


def load_processed_arrays(processed_dir: str, dataset: str, entity: Optional[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset_dir = _safe_join(processed_dir, dataset)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Processed dataset folder does not exist: {dataset_dir}")

    train_path, test_path, labels_path = resolve_entity_triplet(dataset_dir, entity)

    train = np.load(train_path)
    test = np.load(test_path)
    labels = np.load(labels_path)

    if train.ndim == 1:
        train = train[:, None]
    if test.ndim == 1:
        test = test[:, None]
    if labels.ndim == 1:
        labels = labels[:, None]

    if test.shape != labels.shape:
        raise ValueError(f"test and labels shape mismatch: {test.shape} vs {labels.shape}")

    return train.astype(np.float32), test.astype(np.float32), labels.astype(np.float32)


def make_windows(data: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    if window_size < 2:
        raise ValueError("window_size must be >= 2")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if data.shape[0] < window_size:
        raise ValueError(f"data length {data.shape[0]} is smaller than window_size {window_size}")

    windows = [data[start : start + window_size] for start in range(0, data.shape[0] - window_size + 1, stride)]
    return np.stack(windows, axis=0)


def make_window_labels(labels: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    # Score is computed for the final timestamp in each window.
    out = []
    for start in range(0, labels.shape[0] - window_size + 1, stride):
        end_idx = start + window_size - 1
        out.append(float(labels[end_idx].max() > 0))
    return np.asarray(out, dtype=np.int64)


def build_pearson_static_graph(train_data: np.ndarray, topk: int) -> dgl.DGLGraph:
    corr = np.corrcoef(train_data, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, -np.inf)

    n = corr.shape[0]
    k = min(topk, max(n - 1, 1))
    src, dst = [], []
    for i in range(n):
        nn_idx = np.argpartition(-corr[i], kth=k - 1)[:k]
        for j in nn_idx:
            src.append(i)
            dst.append(int(j))
    return dgl.graph((src, dst), num_nodes=n)


def build_model(args, num_vars: int, train_data: np.ndarray, device: torch.device) -> ModelPrototype:
    static_graph = None
    if args.graph_ablation == "pearson_static":
        static_graph = build_pearson_static_graph(train_data, args.topk)

    model = ModelPrototype(
        num_vars=num_vars,
        hidden_dim=args.hidden_dim,
        topk=args.topk,
        graph_ablation=args.graph_ablation,
        graph_update_freq=args.graph_update_freq,
        static_graph=static_graph,
        graph_similarity_metric=args.graph_similarity_metric,
    )
    return model.to(device)


def make_dataloaders(args, train_data: np.ndarray, test_data: np.ndarray, labels: np.ndarray):
    train_windows = make_windows(train_data, args.window_size, args.train_stride)
    test_windows = make_windows(test_data, args.window_size, args.test_stride)
    test_window_labels = make_window_labels(labels, args.window_size, args.test_stride)

    train_loader = DataLoader(
        WindowDataset(train_windows),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        WindowDataset(test_windows),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    # For threshold calibration, keep deterministic order on train windows.
    train_eval_loader = DataLoader(
        WindowDataset(train_windows),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    return train_loader, train_eval_loader, test_loader, test_window_labels


def build_trainer(args, model: ModelPrototype, device: torch.device) -> Tuple[Trainer, torch.optim.Optimizer]:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        w_mse=args.w_mse,
        w_graph_diff=args.w_graph_diff,
        w_graph_sparse=args.w_graph_sparse,
        score_aggregation=args.score_aggregation,
        score_topk_ratio=args.score_topk_ratio,
    )
    return trainer, optimizer


def save_checkpoint(path: str, model: ModelPrototype, optimizer: torch.optim.Optimizer, args, num_vars: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "num_vars": num_vars,
        "model_args": {
            "hidden_dim": args.hidden_dim,
            "topk": args.topk,
            "graph_ablation": args.graph_ablation,
            "graph_update_freq": args.graph_update_freq,
            "graph_similarity_metric": args.graph_similarity_metric,
        },
    }
    torch.save(ckpt, path)


def train_pipeline(args):
    set_seed(args.seed)
    device = choose_device()

    train_data, test_data, labels = load_processed_arrays(args.processed_dir, args.dataset, args.entity)
    num_vars = train_data.shape[1]

    model = build_model(args, num_vars, train_data, device)
    trainer, optimizer = build_trainer(args, model, device)
    train_loader, train_eval_loader, test_loader, test_window_labels = make_dataloaders(args, train_data, test_data, labels)

    history = []
    for epoch in range(1, args.epochs + 1):
        epoch_loss = trainer.train_epoch(train_loader)
        history.append(float(epoch_loss))
        print(f"Epoch {epoch}/{args.epochs} | loss={epoch_loss:.6f}")

    train_scores = trainer.inference(train_eval_loader).numpy()

    run_name = args.run_name
    if args.entity:
        run_name = f"{run_name}_{args.entity}"
    ckpt_path = os.path.join(args.save_dir, f"{run_name}.pt")
    save_checkpoint(ckpt_path, model, optimizer, args, num_vars)

    return {
        "device": device,
        "model": model,
        "trainer": trainer,
        "train_loader": train_loader,
        "train_eval_loader": train_eval_loader,
        "test_loader": test_loader,
        "test_window_labels": test_window_labels,
        "train_scores": train_scores,
        "ckpt_path": ckpt_path,
        "loss_history": history,
    }


def load_checkpoint_for_eval(args):
    if not args.checkpoint_path:
        raise ValueError("--checkpoint_path is required when --mode eval")

    set_seed(args.seed)
    device = choose_device()
    train_data, test_data, labels = load_processed_arrays(args.processed_dir, args.dataset, args.entity)

    ckpt = torch.load(args.checkpoint_path, map_location=device)
    num_vars = int(ckpt["num_vars"])

    model_args = ckpt.get("model_args", {})
    static_graph = None
    if model_args.get("graph_ablation") == "pearson_static":
        static_graph = build_pearson_static_graph(train_data, int(model_args.get("topk", 5)))

    model = ModelPrototype(
        num_vars=num_vars,
        hidden_dim=int(model_args.get("hidden_dim", args.hidden_dim)),
        topk=int(model_args.get("topk", args.topk)),
        graph_ablation=model_args.get("graph_ablation", args.graph_ablation),
        graph_update_freq=int(model_args.get("graph_update_freq", args.graph_update_freq)),
        static_graph=static_graph,
        graph_similarity_metric=model_args.get("graph_similarity_metric", args.graph_similarity_metric),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    trainer, _ = build_trainer(args, model, device)
    _, train_eval_loader, test_loader, test_window_labels = make_dataloaders(args, train_data, test_data, labels)
    train_scores = trainer.inference(train_eval_loader).numpy()

    return {
        "device": device,
        "model": model,
        "trainer": trainer,
        "train_eval_loader": train_eval_loader,
        "test_loader": test_loader,
        "test_window_labels": test_window_labels,
        "train_scores": train_scores,
        "ckpt_path": args.checkpoint_path,
    }
