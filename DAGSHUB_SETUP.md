# DagsHub MLflow tracking setup

The training code supports DagsHub through standard MLflow environment
variables. No DagsHub token is stored in this repository.

Official DagsHub MLflow URI format:

```text
https://dagshub.com/<DagsHub-user-name>/<repository-name>.mlflow
```

For this lab, if your DagsHub username and repository match GitHub, use:

```text
https://dagshub.com/TesWy/2A202600084-HuynhNhutHuy-Day21.mlflow
```

## Local run

In PowerShell, set these variables only in the current terminal:

```powershell
$env:MLFLOW_TRACKING_URI="https://dagshub.com/<USER>/<REPO>.mlflow"
$env:MLFLOW_TRACKING_USERNAME="<DAGSHUB_USERNAME>"
$env:MLFLOW_TRACKING_PASSWORD="<DAGSHUB_TOKEN>"
$env:MLFLOW_EXPERIMENT_NAME="day21-aws-mlops"

.\.venv\Scripts\python.exe src\train.py
```

Then clear the token from the current terminal:

```powershell
Remove-Item Env:\MLFLOW_TRACKING_PASSWORD
```

## GitHub Actions secrets

Add these in GitHub:

```text
Settings -> Secrets and variables -> Actions -> New repository secret
```

Required for DagsHub remote MLflow:

| Secret | Value |
|---|---|
| `MLFLOW_TRACKING_URI` | `https://dagshub.com/<USER>/<REPO>.mlflow` |
| `MLFLOW_TRACKING_USERNAME` | DagsHub username |
| `MLFLOW_TRACKING_PASSWORD` | DagsHub access token |
| `MLFLOW_EXPERIMENT_NAME` | `day21-aws-mlops` |

The workflow already exports these secrets before running `python src/train.py`.

## What to screenshot

After a GitHub Actions run completes, open the DagsHub repository and go to the
Experiments/MLflow UI. Screenshot a run logged from GitHub Actions showing:

- parameters such as `model_type`, `train_rows`, `eval_rows`
- metrics such as `accuracy`, `f1_score`, `precision_weighted`, `recall_weighted`
- the run timestamp after the CI/CD run
