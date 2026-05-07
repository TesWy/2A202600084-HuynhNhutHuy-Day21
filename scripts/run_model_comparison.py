import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.train import train


DEFAULT_CONFIGS = [
    "configs/logistic_regression.yaml",
    "configs/gradient_boosting.yaml",
    "configs/mlp.yaml",
    "configs/random_forest_strong.yaml",
]


def _load_params(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_metrics() -> dict:
    with open("outputs/metrics.json", encoding="utf-8") as f:
        return json.load(f)


def _write_comparison(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "model_comparison.json"
    csv_path = output_dir / "model_comparison.csv"
    md_path = output_dir / "model_comparison.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    fieldnames = [
        "run_name",
        "config",
        "model_type",
        "accuracy",
        "f1_score",
        "precision_weighted",
        "recall_weighted",
        "train_rows",
        "data_dvc_md5",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fieldnames} for row in rows])

    lines = [
        "# Model Comparison",
        "",
        "| Run | Model | Accuracy | F1 weighted | Train rows | Data DVC md5 |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {run_name} | {model_type} | {accuracy:.4f} | {f1_score:.4f} | "
            "{train_rows} | `{data_dvc_md5}` |".format(**row)
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multiple model configs and write a comparison table."
    )
    parser.add_argument("--data-path", default="data/train_phase1.csv")
    parser.add_argument("--eval-path", default="data/eval.csv")
    parser.add_argument("--output-dir", default="outputs/evidence")
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        help="Config YAML path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--include-lnn",
        action="store_true",
        help="Also run configs/lnn.yaml if torch and ncps are installed.",
    )
    args = parser.parse_args()

    config_paths = [Path(path) for path in (args.configs or DEFAULT_CONFIGS)]
    if args.include_lnn:
        config_paths.append(Path("configs/lnn.yaml"))

    rows = []
    for config_path in config_paths:
        params = _load_params(config_path)
        run_name = f"compare_{config_path.stem}"
        print(f"Running {run_name} from {config_path}")
        try:
            train(params, data_path=args.data_path, eval_path=args.eval_path, run_name=run_name)
        except ImportError as exc:
            if params.get("model_type") == "lnn":
                print(f"Skipping {run_name}: {exc}")
                continue
            raise

        metrics = _read_metrics()
        metrics["run_name"] = run_name
        metrics["config"] = str(config_path)
        rows.append(metrics)

    rows.sort(key=lambda row: row["f1_score"], reverse=True)
    _write_comparison(rows, Path(args.output_dir))

    best = rows[0]
    print(
        "Best run: {run_name} | model={model_type} | accuracy={accuracy:.4f} | "
        "f1={f1_score:.4f}".format(**best)
    )


if __name__ == "__main__":
    main()
