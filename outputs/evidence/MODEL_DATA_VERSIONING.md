# Model and Data Version Evidence

## Current status before final versioning run

S3 production model currently contains only the mutable production pointer:

```text
models/latest/model.pkl
models/latest/metrics.json
models/latest/report.txt
```

The workflow has been updated so the next successful deploy will also publish:

```text
models/runs/<git_sha>/model.pkl
models/runs/<git_sha>/metrics.json
models/runs/<git_sha>/report.txt
```

## Data versions

Data is already versioned through Git plus DVC.

| Data version | Commit | DVC md5 | Size |
| --- | --- | --- | --- |
| Phase 1 only | `f9fb17c` | `c43afab731fd6431a94f888fdc687876` | `184090` bytes |
| Phase 1 + Phase 2 | `30ff7ca` and current | `5853e7711c78f02286e65fca6cb6e124` | `368068` bytes |

The row-level difference is expected:

- original `train_phase1.csv`: 2998 rows
- after adding `train_phase2.csv`: 5996 rows

## Model versions after next green CI run

Check with:

```bash
aws s3 ls s3://teswy-2a202600084-day21-mlops/models/ --recursive
```

Expected evidence:

- `models/latest/...` proves what EC2 currently serves.
- `models/runs/<git_sha>/...` proves immutable historical model versions.
- `metrics.json` contains `git_sha`, `github_run_id`, `train_rows`, accuracy, and F1.
