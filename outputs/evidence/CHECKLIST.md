# Evidence checklist

## Step 1 - MLflow local experiments

Status: DONE

Evidence files:
- outputs/mlflow_step1_wide.png
- outputs/mlflow_step1_metrics_summary.png
- outputs/metrics.json
- outputs/report.txt

Best Step 1 model on train_phase1:
- model_type: random_forest
- accuracy: 0.6920
- f1_score: 0.6912

## Step 2 - DVC + AWS + CI/CD

Status: FUNCTIONALLY DONE

Completed:
- DVC initialized
- Data tracked with DVC pointer files
- DVC remote configured: s3://teswy-2a202600084-day21-mlops/dvc
- DVC push succeeded
- S3 contains DVC objects and production model artifacts
- EC2 instance is running
- FastAPI service is running on EC2
- /health works
- /predict works
- First GitHub Actions run proves eval gate blocks deploy when accuracy < 0.70
- Second GitHub Actions run is all green: Unit Test, Train, Eval, Deploy
- Versioning GitHub Actions run is all green: Unit Test, Train, Eval, Deploy

Important nuance:
- The all-green run is triggered by the Step 3 data commit: data: add phase2 training samples.
- This is good evidence for continuous training and deploy, but keep the first failed run as eval-gate evidence.

Evidence files:
- outputs/evidence/s3_listing.txt
- outputs/evidence/s3_models_latest_after_green_deploy.txt
- outputs/evidence/ec2_instance.json
- outputs/evidence/api_health.json
- outputs/evidence/api_predict.json
- outputs/evidence/api_health_after_green_deploy.json
- outputs/evidence/api_predict_after_green_deploy.json
- outputs/evidence/production_metrics_after_green_deploy.json
- outputs/evidence/github_actions_runs.json
- outputs/evidence/github_actions_jobs_run1.json
- outputs/evidence/github_actions_jobs_run2_data_update.json
- outputs/evidence/github_actions_jobs_versioned_model_run.json
- outputs/evidence/s3_models_versioned_after_green_deploy.txt
- outputs/evidence/versioned_model_metrics_0e13abf.json
- outputs/evidence/api_health_after_versioned_deploy.json
- outputs/evidence/api_predict_after_versioned_deploy.json

Latest production metrics after green deploy:
- accuracy: 0.7500
- f1_score: 0.7491
- git_sha: 0e13abfe471a9daddc9becb361e30574e7476488
- train_rows: 5996

Manual screenshots still recommended:
- GitHub Actions failed eval gate run
- GitHub Actions all-green data update run
- S3 console showing dvc/ objects
- S3 console showing models/latest/model.pkl
- EC2 console showing mlops-serve-day21 running
- Browser/terminal output for /health and /predict
- DagsHub Experiments tab showing multiple runs and accuracy/f1_score columns
- S3 console showing models/runs/0e13abfe471a9daddc9becb361e30574e7476488/

## Step 3 - Continuous training

Status: DONE

Completed:
- data/train_phase1.csv updated from 2998 to 5996 rows by adding train_phase2
- DVC pointer data/train_phase1.csv.dvc updated
- DVC push completed before git push
- Git commit pushed: data: add phase2 training samples
- GitHub Actions automatically retrained and deployed

## Bonus status

Implemented in code/workflow:
- Multi-model training: logistic_regression, random_forest, extra_trees, gradient_boosting, mlp, optional lnn
- Automated report: outputs/report.txt
- Label distribution drift warning: metrics.json includes label_distribution and drift_warnings
- Rollback guard: workflow compares new accuracy with previous production metrics
- DagsHub remote MLflow tracking: GitHub Actions can log experiments to DagsHub when secrets are configured
- Data versioning evidence: Git + DVC md5 changed from phase 1 to phase 2 data
- Model versioning evidence: workflow now publishes both models/latest and models/runs/<git_sha>

Not fully complete yet:
- Capture screenshot of DagsHub Experiments tab showing accuracy/f1_score columns
- LNN experiment is optional and requires pip install -r requirements-lnn.txt
