# AWS setup notes

This project uses AWS equivalents for the original lab cloud resources:

- Object storage: Amazon S3
- Virtual machine: Amazon EC2
- Data versioning remote: DVC S3 remote
- Serving runtime: FastAPI systemd service on EC2

## Required GitHub Secrets

Create these in GitHub: Settings > Secrets and variables > Actions.

| Secret | Meaning |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM access key with S3 object read/write access |
| `AWS_SECRET_ACCESS_KEY` | IAM secret access key |
| `AWS_REGION` | AWS region, for example `ap-southeast-1` |
| `AWS_S3_BUCKET` | S3 bucket name |
| `VM_HOST` | EC2 public IPv4 address or DNS name |
| `VM_USER` | EC2 SSH user, usually `ubuntu` for Ubuntu AMIs |
| `VM_SSH_KEY` | Private SSH key allowed to connect to the EC2 instance |
| `MLFLOW_TRACKING_URI` | Optional DagsHub/remote MLflow URI |
| `MLFLOW_TRACKING_USERNAME` | Optional remote MLflow username |
| `MLFLOW_TRACKING_PASSWORD` | Optional remote MLflow token/password |
| `MLFLOW_EXPERIMENT_NAME` | Optional MLflow experiment name, for example `day21-aws-mlops` |

## DVC remote

Run locally after creating your S3 bucket:

```powershell
dvc init
dvc remote add -d myremote s3://<YOUR_BUCKET_NAME>/dvc
dvc add data/train_phase1.csv
dvc add data/eval.csv
dvc add data/train_phase2.csv
dvc push
```

Commit only the DVC pointer files, not the CSV files:

```powershell
git add .dvc/config data/*.dvc .gitignore
git commit -m "feat: track datasets with DVC on S3"
```

## EC2 service

Current lab resources created for this repository:

| Resource | Value |
|---|---|
| Region | `us-east-1` |
| S3 bucket | `teswy-2a202600084-day21-mlops` |
| DVC remote | `s3://teswy-2a202600084-day21-mlops/dvc` |
| EC2 instance id | `i-03ea0a18db56e45ff` |
| EC2 public IP | `44.204.117.234` |
| EC2 user | `ubuntu` |
| Security group | `sg-0db90d2e035790962` |
| Service name | `mlops-serve` |

Smoke tests:

```powershell
curl http://44.204.117.234:8000/health

curl -X POST http://44.204.117.234:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"features": [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0]}'
```

On EC2, install runtime dependencies:

```bash
sudo apt update
sudo apt install -y python3-pip
pip3 install fastapi uvicorn scikit-learn joblib boto3 pandas
mkdir -p ~/models ~/src
```

Copy `src/serve.py` to `~/src/serve.py` on EC2.

Create `/etc/systemd/system/mlops-serve.service`:

```ini
[Unit]
Description=MLOps Model Inference Server
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu
Environment="S3_BUCKET=<YOUR_BUCKET_NAME>"
Environment="AWS_DEFAULT_REGION=<YOUR_REGION>"
ExecStart=/usr/bin/python3 /home/ubuntu/src/serve.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

If you use IAM role on EC2, no key file is needed on the instance. If you use an IAM
user instead, configure AWS credentials on EC2 with `aws configure` or environment
variables.

Enable the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable mlops-serve
```

Open inbound TCP port `8000` in the EC2 security group for testing.
