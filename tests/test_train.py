import json
import os

import numpy as np
import pandas as pd

from src.train import build_model, train


FEATURE_NAMES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "wine_type",
]


def _make_temp_data(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    X = rng.random((n, len(FEATURE_NAMES)))
    y = rng.integers(0, 3, size=n)

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y

    train_path = str(tmp_path / "train.csv")
    eval_path = str(tmp_path / "eval.csv")
    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path, index=False)
    return train_path, eval_path


def test_build_model_supports_required_model_types():
    for model_type in [
        "logistic_regression",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "mlp",
    ]:
        model = build_model({"model_type": model_type, "n_estimators": 10, "max_iter": 20})
        assert model is not None


def test_train_returns_float(tmp_path):
    train_path, eval_path = _make_temp_data(tmp_path)
    acc = train(
        {"model_type": "random_forest", "n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_metrics_file_created(tmp_path):
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"model_type": "random_forest", "n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("outputs/metrics.json")
    with open("outputs/metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert "precision_weighted" in metrics
    assert "recall_weighted" in metrics
    assert "confusion_matrix" in metrics
    assert "label_distribution" in metrics


def test_report_and_model_files_created(tmp_path):
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"model_type": "random_forest", "n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("outputs/report.txt")
    assert os.path.exists("models/model.pkl")
