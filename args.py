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

    parser.add_argument(
        "--model_name",
        type=str,
        default="hyper_tsad",
        choices=["hyper_tsad", "tranad", "anomaly_transformer"],
        help="Model family to train/evaluate.",
    )

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
    parser.add_argument(
        "--hypergraph_encoder_type",
        type=str,
        default="conv",
        choices=["conv", "attn"],
        help="Hypergraph encoder backend used when graph_ablation=dynamic_hypergraph.",
    )
    parser.add_argument("--hypergraph_attn_heads", type=int, default=4)
    parser.add_argument("--hypergraph_attn_dropout", type=float, default=0.1)

    # TranAD baseline params
    parser.add_argument("--tranad_d_ff", type=int, default=256)
    parser.add_argument("--tranad_dropout", type=float, default=0.1)

    # Anomaly Transformer baseline params
    parser.add_argument("--at_d_model", type=int, default=128)
    parser.add_argument("--at_n_heads", type=int, default=8)
    parser.add_argument("--at_e_layers", type=int, default=3)
    parser.add_argument("--at_d_ff", type=int, default=256)
    parser.add_argument("--at_dropout", type=float, default=0.1)
    parser.add_argument("--at_activation", type=str, default="gelu", choices=["gelu", "relu"])

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

    parser.add_argument(
        "--threshold_method",
        type=str,
        default="pot",
        choices=["quantile", "pot"],
        help="Threshold search method for anomaly decision.",
    )
    parser.add_argument("--threshold_quantile", type=float, default=0.99)
    parser.add_argument("--pot_init_level", type=float, default=0.98)
    parser.add_argument("--pot_risk", type=float, default=1e-3)
    parser.add_argument("--score_normalize", action="store_true", default=False,
                        help="Normalise per-variable residuals by calibration stats before aggregation.")
    parser.add_argument("--score_horizons", type=int, default=3,
                        help="Number of prediction horizons to average for anomaly score (>=1).")
    parser.add_argument("--seed", type=int, default=42)

    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()
