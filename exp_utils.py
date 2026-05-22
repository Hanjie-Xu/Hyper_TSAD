"""Experiment folder and naming utilities."""

import json
import os
from datetime import datetime
from typing import Any, Dict


def generate_exp_name(args: Any) -> str:
    """Generate experiment name from args: <model>-<dataset>-<timestamp>.

    Args:
        args: ArgumentParser namespace with attributes 'dataset', 'entity',
              'graph_ablation', 'hypergraph_encoder_type'.
    Returns:
        Experiment name string, e.g., "dynamic_hypergraph-attn-SMD-machine-1-1-20260521_143022"
    """
    dataset_part = args.dataset
    if args.entity:
        dataset_part = f"{args.dataset}_{args.entity}"

    model_name = getattr(args, "model_name", "hyper_tsad")

    if model_name == "hyper_tsad":
        model_part = args.graph_ablation
        if args.graph_ablation == "dynamic_hypergraph":
            model_part = f"{model_part}-{args.hypergraph_encoder_type}"
    else:
        model_part = model_name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_part}-{dataset_part}-{timestamp}"


def create_exp_structure(exp_name: str, base_dir: str = "experiments") -> Dict[str, str]:
    """Create experiment folder structure and return paths to key folders/files.

    Structure:
        experiments/<exp_name>/
        ├── args.json           (all CLI arguments)
        ├── loss.json           (per-epoch loss list)
        ├── model.pt            (checkpoint)
        └── plots/              (subfolder for all plots)

    Args:
        exp_name: Experiment name (usually from generate_exp_name).
        base_dir: Base directory for all experiments.
    Returns:
        Dict with keys: 'root', 'args_path', 'loss_path', 'model_path', 'plots_dir'
    """
    exp_root = os.path.join(base_dir, exp_name)
    os.makedirs(exp_root, exist_ok=True)

    plots_dir = os.path.join(exp_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    return {
        "root": exp_root,
        "args_path": os.path.join(exp_root, "args.json"),
        "loss_path": os.path.join(exp_root, "loss.json"),
        "model_path": os.path.join(exp_root, "model.pt"),
        "plots_dir": plots_dir,
    }


def save_args_json(args: Any, path: str) -> None:
    """Save all CLI arguments to a JSON file."""
    args_dict = vars(args)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=2, default=str)


def save_loss_json(loss_history: list, path: str) -> None:
    """Save loss history to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(loss_history, f, indent=2)


def load_args_json(path: str) -> Dict[str, Any]:
    """Load CLI arguments from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
