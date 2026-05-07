import argparse
import json
import os

import boto3
from botocore.exceptions import ClientError


LATEST_KEYS = {
    "model.pkl": "models/latest/model.pkl",
    "metrics.json": "models/latest/metrics.json",
    "report.txt": "models/latest/report.txt",
}


def _read_metrics(s3_client, bucket: str, run_prefix: str) -> dict:
    obj = s3_client.get_object(Bucket=bucket, Key=f"{run_prefix}/metrics.json")
    return json.loads(obj["Body"].read())


def promote_model_version(bucket: str, git_sha: str, dry_run: bool = False) -> None:
    s3 = boto3.client("s3")
    run_prefix = f"models/runs/{git_sha}"

    try:
        metrics = _read_metrics(s3, bucket, run_prefix)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"NoSuchKey", "404"}:
            raise SystemExit(f"No model version found at s3://{bucket}/{run_prefix}") from exc
        raise

    print("Candidate model version:")
    print(f"- git_sha: {metrics.get('git_sha', git_sha)}")
    print(f"- model_type: {metrics.get('model_type')}")
    print(f"- train_rows: {metrics.get('train_rows')}")
    print(f"- accuracy: {metrics.get('accuracy')}")
    print(f"- f1_score: {metrics.get('f1_score')}")

    for file_name, latest_key in LATEST_KEYS.items():
        source_key = f"{run_prefix}/{file_name}"
        copy_source = {"Bucket": bucket, "Key": source_key}
        print(f"{'Would promote' if dry_run else 'Promoting'} {source_key} -> {latest_key}")
        if not dry_run:
            s3.copy_object(Bucket=bucket, CopySource=copy_source, Key=latest_key)

    if not dry_run:
        print("Promotion complete. Restart the EC2 service so it reloads models/latest/model.pkl.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote or roll back an immutable S3 model version to models/latest."
    )
    parser.add_argument("git_sha", help="Full Git SHA under models/runs/<git_sha>.")
    parser.add_argument(
        "--bucket",
        default=os.environ.get("S3_BUCKET"),
        help="S3 bucket name. Defaults to S3_BUCKET env var.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without copying.")
    args = parser.parse_args()

    if not args.bucket:
        raise SystemExit("Missing bucket. Pass --bucket or set S3_BUCKET.")

    promote_model_version(args.bucket, args.git_sha, args.dry_run)


if __name__ == "__main__":
    main()
