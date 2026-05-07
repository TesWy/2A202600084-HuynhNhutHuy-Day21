# Data and Model Versioning

This project tracks data, model artifacts, and experiment metrics as separate
versioned assets.

## Data Versions

Training data is versioned by Git plus DVC pointer files. The large CSV stays
outside Git, while `data/train_phase1.csv.dvc` records the exact object hash
stored in the DVC remote on S3.

| Stage | Git commit | DVC md5 | Size |
| --- | --- | --- | --- |
| Initial training data | `f9fb17c` | `c43afab731fd6431a94f888fdc687876` | `184090` bytes |
| Phase 2 training data | `30ff7ca` and later | `5853e7711c78f02286e65fca6cb6e124` | `368068` bytes |

To inspect the data version for a commit:

```bash
git show <commit>:data/train_phase1.csv.dvc
dvc pull data/train_phase1.csv.dvc
```

## Model Versions

The CI/CD pipeline publishes two model locations to S3 after the Eval gate
passes:

| Location | Purpose |
| --- | --- |
| `s3://teswy-2a202600084-day21-mlops/models/latest/model.pkl` | Mutable production pointer used by the EC2 API |
| `s3://teswy-2a202600084-day21-mlops/models/runs/<git_sha>/model.pkl` | Immutable model version for one GitHub Actions run |

Each model version also includes:

- `metrics.json`
- `report.txt`

The `metrics.json` file records model, run, and data metadata:

- `model_type`
- `git_sha`
- `github_run_id`
- `github_run_attempt`
- `data_path`
- `eval_path`
- `train_rows`
- `eval_rows`
- `accuracy`
- `f1_score`
- `precision_weighted`
- `recall_weighted`

## Experiment Versions

MLflow runs are logged to DagsHub when these GitHub Actions secrets are present:

- `MLFLOW_TRACKING_URI`
- `MLFLOW_TRACKING_USERNAME`
- `MLFLOW_TRACKING_PASSWORD`
- `MLFLOW_EXPERIMENT_NAME`

DagsHub is used to compare experiment runs by metrics such as accuracy and
weighted F1. S3 is used for production model artifacts and EC2 deployment.
