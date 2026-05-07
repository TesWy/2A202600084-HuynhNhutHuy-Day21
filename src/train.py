import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EVAL_THRESHOLD = 0.70
FEATURE_COUNT = 12
LABELS = [0, 1, 2]


class TorchCfCClassifier(BaseEstimator, ClassifierMixin):
    """
    Small optional Liquid-style classifier using ncps.torch.CfC.

    Wine Quality is tabular, not a natural sequence task. For experimentation we
    treat the 12 features as a short sequence of scalar observations. Keep this
    model out of the default CI path unless torch and ncps are installed.
    """

    def __init__(
        self,
        hidden_units=32,
        epochs=30,
        lr=0.01,
        batch_size=64,
        random_state=42,
    ):
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y):
        try:
            import numpy as np
            import torch
            from ncps.torch import CfC
        except ImportError as exc:
            raise ImportError(
                "model_type='lnn' requires optional packages: pip install torch ncps"
            ) from exc

        torch.manual_seed(self.random_state)
        X_np = X.to_numpy(dtype="float32") if hasattr(X, "to_numpy") else X.astype("float32")
        y_np = y.to_numpy(dtype="int64") if hasattr(y, "to_numpy") else y.astype("int64")
        self.classes_ = np.array(sorted(set(y_np.tolist())))

        X_tensor = torch.tensor(X_np, dtype=torch.float32).unsqueeze(-1)
        y_tensor = torch.tensor(y_np, dtype=torch.long)

        self.network_ = CfC(
            input_size=1,
            units=self.hidden_units,
            proj_size=len(self.classes_),
            batch_first=True,
            return_sequences=False,
        )
        optimizer = torch.optim.Adam(self.network_.parameters(), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state),
        )

        self.network_.train()
        for _ in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                logits, _ = self.network_(batch_X)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        import torch

        X_np = X.to_numpy(dtype="float32") if hasattr(X, "to_numpy") else X.astype("float32")
        X_tensor = torch.tensor(X_np, dtype=torch.float32).unsqueeze(-1)
        self.network_.eval()
        with torch.no_grad():
            logits, _ = self.network_(X_tensor)
            pred_idx = logits.argmax(dim=1).cpu().numpy()
        return self.classes_[pred_idx]


def _clean_params(params: dict) -> dict:
    cleaned = dict(params or {})
    cleaned.setdefault("model_type", "random_forest")
    return cleaned


def build_model(params: dict):
    model_type = params["model_type"]

    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=params.get("max_depth", 5),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", "sqrt"),
            random_state=42,
            class_weight=params.get("class_weight"),
            n_jobs=int(params.get("n_jobs", -1)),
        )

    if model_type == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=int(params.get("n_estimators", 500)),
            max_depth=params.get("max_depth"),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", "sqrt"),
            random_state=42,
            class_weight=params.get("class_weight", "balanced"),
            n_jobs=int(params.get("n_jobs", -1)),
        )

    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=int(params.get("max_depth", 3)),
            random_state=42,
        )

    if model_type == "logistic_regression":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=float(params.get("C", 1.0)),
                        max_iter=int(params.get("max_iter", 1000)),
                        class_weight=params.get("class_weight"),
                        random_state=42,
                    ),
                ),
            ]
        )

    if model_type == "mlp":
        hidden_layer_sizes = params.get("hidden_layer_sizes", [64, 32])
        if isinstance(hidden_layer_sizes, int):
            hidden_layer_sizes = [hidden_layer_sizes]
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=tuple(hidden_layer_sizes),
                        alpha=float(params.get("alpha", 0.0001)),
                        learning_rate_init=float(params.get("learning_rate_init", 0.001)),
                        max_iter=int(params.get("max_iter", 500)),
                        early_stopping=True,
                        random_state=42,
                    ),
                ),
            ]
        )

    if model_type == "lnn":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    TorchCfCClassifier(
                        hidden_units=int(params.get("hidden_units", 32)),
                        epochs=int(params.get("epochs", 30)),
                        lr=float(params.get("learning_rate", 0.01)),
                        batch_size=int(params.get("batch_size", 64)),
                        random_state=42,
                    ),
                ),
            ]
        )

    raise ValueError(
        "Unsupported model_type. Use one of: random_forest, extra_trees, gradient_boosting, "
        "logistic_regression, mlp, lnn."
    )


def _label_distribution(y_train) -> dict:
    counts = y_train.value_counts(normalize=True).to_dict()
    return {str(label): float(counts.get(label, 0.0)) for label in LABELS}


def _dvc_metadata(data_path: str) -> dict:
    dvc_path = Path(f"{data_path}.dvc")
    if not dvc_path.exists():
        return {"md5": None, "size": None}

    with open(dvc_path, encoding="utf-8") as f:
        dvc_info = yaml.safe_load(f) or {}

    outs = dvc_info.get("outs") or [{}]
    first_out = outs[0]
    return {
        "md5": first_out.get("md5"),
        "size": first_out.get("size"),
    }


def _write_outputs(metrics: dict, report_text: str, model) -> None:
    Path("outputs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    with open("outputs/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open("outputs/report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    joblib.dump(model, "models/model.pkl")


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
    run_name: str | None = None,
) -> float:
    params = _clean_params(params)

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)
    data_dvc = _dvc_metadata(data_path)
    eval_dvc = _dvc_metadata(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    if X_train.shape[1] != FEATURE_COUNT or X_eval.shape[1] != FEATURE_COUNT:
        raise ValueError(f"Expected {FEATURE_COUNT} features for Wine Quality.")

    model = build_model(params)
    label_distribution = _label_distribution(y_train)
    drift_warnings = [
        f"class {label} ratio {ratio:.3f} is below 0.10"
        for label, ratio in label_distribution.items()
        if ratio < 0.10
    ]

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_params(
            {
                "data_path": data_path,
                "eval_path": eval_path,
                "train_rows": len(df_train),
                "eval_rows": len(df_eval),
            }
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_eval)

        acc = float(accuracy_score(y_eval, preds))
        f1 = float(f1_score(y_eval, preds, average="weighted", zero_division=0))
        precision = float(precision_score(y_eval, preds, average="weighted", zero_division=0))
        recall = float(recall_score(y_eval, preds, average="weighted", zero_division=0))

        report_text = classification_report(
            y_eval,
            preds,
            labels=LABELS,
            target_names=["low", "medium", "high"],
            zero_division=0,
        )
        matrix = confusion_matrix(y_eval, preds, labels=LABELS).tolist()

        metrics = {
            "model_type": params["model_type"],
            "git_sha": os.environ.get("GITHUB_SHA", "local"),
            "github_run_id": os.environ.get("GITHUB_RUN_ID", "local"),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "local"),
            "data_path": data_path,
            "eval_path": eval_path,
            "train_rows": len(df_train),
            "eval_rows": len(df_eval),
            "data_dvc_md5": data_dvc["md5"],
            "data_dvc_size": data_dvc["size"],
            "eval_dvc_md5": eval_dvc["md5"],
            "eval_dvc_size": eval_dvc["size"],
            "accuracy": acc,
            "f1_score": f1,
            "precision_weighted": precision,
            "recall_weighted": recall,
            "confusion_matrix": matrix,
            "label_distribution": label_distribution,
            "drift_warnings": drift_warnings,
            "eval_threshold": EVAL_THRESHOLD,
        }

        for key in ["accuracy", "f1_score", "precision_weighted", "recall_weighted"]:
            mlflow.log_metric(key, metrics[key])
        for label, ratio in label_distribution.items():
            mlflow.log_metric(f"train_label_ratio_{label}", ratio)
        mlflow.set_tags(
            {
                "git_sha": metrics["git_sha"],
                "github_run_id": metrics["github_run_id"],
                "run_name": run_name or params["model_type"],
                "data_path": data_path,
                "eval_path": eval_path,
                "data_dvc_md5": data_dvc["md5"],
                "eval_dvc_md5": eval_dvc["md5"],
            }
        )

        _write_outputs(metrics, report_text, model)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact("outputs/metrics.json")
        mlflow.log_artifact("outputs/report.txt")

        if drift_warnings:
            print("Data distribution warning:")
            for warning in drift_warnings:
                print(f"- {warning}")
        print(f"Model: {params['model_type']} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

    return acc


if __name__ == "__main__":
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")
    if experiment_name:
        mlflow.set_experiment(experiment_name)

    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    train(params)
