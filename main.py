import json
import os

from args import parse_args
from evaluation import evaluate_model
from training import load_checkpoint_for_eval, train_pipeline


def _to_printable_metrics(metrics):
    out = {}
    for k, v in metrics.items():
        if isinstance(v, float):
            if v != v:
                out[k] = None
            else:
                out[k] = round(v, 6)
        else:
            out[k] = v
    return out


def main() -> None:
    args = parse_args()

    if args.mode in {"train", "both"}:
        run = train_pipeline(args)
        print(f"Training done. Checkpoint saved to: {run['ckpt_path']}")

        if args.mode == "both":
            metrics, _ = evaluate_model(
                trainer=run["trainer"],
                test_loader=run["test_loader"],
                test_window_labels=run["test_window_labels"],
                train_scores=run["train_scores"],
                threshold_quantile=args.threshold_quantile,
            )
            printable = _to_printable_metrics(metrics)
            print("Evaluation metrics:")
            print(json.dumps(printable, indent=2))

            metrics_path = os.path.join(args.save_dir, f"{args.run_name}_metrics.json")
            os.makedirs(args.save_dir, exist_ok=True)
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(printable, f, indent=2)
            print(f"Saved metrics to: {metrics_path}")

    elif args.mode == "eval":
        run = load_checkpoint_for_eval(args)
        metrics, _ = evaluate_model(
            trainer=run["trainer"],
            test_loader=run["test_loader"],
            test_window_labels=run["test_window_labels"],
            train_scores=run["train_scores"],
            threshold_quantile=args.threshold_quantile,
        )
        printable = _to_printable_metrics(metrics)
        print("Evaluation metrics:")
        print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
