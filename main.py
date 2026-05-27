import json
import os
import time
from datetime import datetime, timezone

from args import parse_args
from evaluation import evaluate_model
from plot import plot_all
from training import load_checkpoint_for_eval, load_processed_arrays, train_pipeline


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


def _resolve_tests_jsonl_path(args, run, exp_paths):
    if exp_paths and exp_paths.get("tests_path"):
        return exp_paths["tests_path"]

    ckpt_path = None
    if isinstance(run, dict):
        ckpt_path = run.get("ckpt_path")
    if not ckpt_path:
        ckpt_path = getattr(args, "checkpoint_path", None)

    if ckpt_path:
        return os.path.join(os.path.dirname(ckpt_path), "tests.jsonl")
    return os.path.join(args.save_dir, "tests.jsonl")


def _append_test_record(path, testing_time_sec, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    record = {
        "tested_at_utc": datetime.now(timezone.utc).isoformat(),
        "testing_time_sec": round(float(testing_time_sec), 6),
        "evaluation_metrics": metrics,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    if args.mode in {"train", "both"}:
        run = train_pipeline(args)
        print(f"Training done. Checkpoint saved to: {run['ckpt_path']}")
        exp_paths = run.get("exp_paths", {})

        if args.mode == "both":
            eval_start = time.perf_counter()
            metrics, test_scores = evaluate_model(
                trainer=run["trainer"],
                test_loader=run["test_loader"],
                test_window_labels=run["test_window_labels"],
                train_scores=run["train_scores"],
                threshold_method=args.threshold_method,
                threshold_quantile=args.threshold_quantile,
                pot_init_level=args.pot_init_level,
                pot_risk=args.pot_risk,
            )
            testing_time_sec = time.perf_counter() - eval_start
            printable = _to_printable_metrics(metrics)
            print("Evaluation metrics:")
            print(json.dumps(printable, indent=2))

            tests_path = _resolve_tests_jsonl_path(args, run, exp_paths)
            _append_test_record(tests_path, testing_time_sec, printable)
            print(f"Test record appended to: {tests_path}")

            # Load raw test data for the sensor-overlay panel.
            _, test_data, _ = load_processed_arrays(
                args.processed_dir, args.dataset, args.entity
            )
            
            # Use experiment-managed plots directory if available
            plots_dir = exp_paths.get("plots_dir", os.path.join(args.save_dir, "plots"))
            plot_all(
                loss_history=run["loss_history"],
                test_scores=test_scores,
                test_labels=run["test_window_labels"],
                threshold=float(metrics["threshold"]),
                test_data=test_data,
                var_index=0,
                title_prefix=args.dataset,
                save_dir=plots_dir,
                run_name=args.run_name,
            )
            
            print(f"Plots saved to: {plots_dir}")
            if exp_paths:
                print(f"Experiment structure:")
                print(f"  Root: {exp_paths.get('root')}")
                print(f"  Args: {exp_paths.get('args_path')}")
                print(f"  Loss: {exp_paths.get('loss_path')}")
                print(f"  Tests: {exp_paths.get('tests_path')}")
                print(f"  Model: {exp_paths.get('model_path')}")
                print(f"  Plots: {exp_paths.get('plots_dir')}")

    elif args.mode == "eval":
        run = load_checkpoint_for_eval(args)
        eval_start = time.perf_counter()
        metrics, test_scores = evaluate_model(
            trainer=run["trainer"],
            test_loader=run["test_loader"],
            test_window_labels=run["test_window_labels"],
            train_scores=run["train_scores"],
            threshold_method=args.threshold_method,
            threshold_quantile=args.threshold_quantile,
            pot_init_level=args.pot_init_level,
            pot_risk=args.pot_risk,
        )
        testing_time_sec = time.perf_counter() - eval_start
        printable = _to_printable_metrics(metrics)
        print("Evaluation metrics:")
        print(json.dumps(printable, indent=2))

        tests_path = _resolve_tests_jsonl_path(args, run, exp_paths={})
        _append_test_record(tests_path, testing_time_sec, printable)
        print(f"Test record appended to: {tests_path}")

        _, test_data, _ = load_processed_arrays(
            args.processed_dir, args.dataset, args.entity
        )
        plot_all(
            loss_history=[],
            test_scores=test_scores,
            test_labels=run["test_window_labels"],
            threshold=float(metrics["threshold"]),
            test_data=test_data,
            var_index=0,
            title_prefix=args.dataset,
            save_dir=os.path.join(args.save_dir, "plots"),
            run_name=args.run_name,
        )


if __name__ == "__main__":
    main()
