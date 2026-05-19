import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate Hyper_TSAD model")

    parser.add_argument("--mode", type=str, default="both", choices=["train", "eval", "both"])
    parser.add_argument("--dataset", type=str, required=True, help="Processed dataset folder name, e.g. SWaT")
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="Entity/channel prefix for multi-entity datasets (e.g. machine-1-1 for SMD, C-1 for MSL)",
    )
    parser.add_argument("--processed_dir", type=str, default="processed")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--run_name", type=str, default="hyper_tsad_run")
    parser.add_argument("--checkpoint_path", type=str, default=None)

    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--train_stride", type=int, default=1)
    parser.add_argument("--test_stride", type=int, default=1)

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--graph_ablation",
        type=str,
        default="dynamic",
        choices=[
            "dynamic",
            "dynamic_hypergraph",
            "pearson_static",
            "identity",
            "fully_connected",
            "none",
        ],
    )
    parser.add_argument("--graph_update_freq", type=int, default=1)
    parser.add_argument(
        "--graph_similarity_metric",
        type=str,
        default="dot_product",
        choices=["dot_product", "cosine"],
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--w_mse", type=float, default=1.0)
    parser.add_argument("--w_graph_diff", type=float, default=0.01)
    parser.add_argument("--w_graph_sparse", type=float, default=0.01)
    parser.add_argument("--score_aggregation", type=str, default="topk", choices=["mean", "topk"])
    parser.add_argument("--score_topk_ratio", type=float, default=0.2)

    parser.add_argument("--threshold_quantile", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)

    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()
