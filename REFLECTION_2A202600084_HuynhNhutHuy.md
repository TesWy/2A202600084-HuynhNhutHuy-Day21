# Reflection - Day 21 CI/CD for AI Systems

**Học viên:** Huỳnh Nhựt Huy  
**Mã học viên:** 2A202600084  
**GitHub repo:** https://github.com/TesWy/2A202600084-HuynhNhutHuy-Day21

## Những gì đã thực hiện

Trong bài lab này, em đã xây dựng một pipeline MLOps hoàn chỉnh cho bài toán phân loại Wine Quality. Ở bước đầu, em hoàn thiện `src/train.py` để huấn luyện mô hình, tính các chỉ số `accuracy`, `f1_score`, `precision`, `recall`, confusion matrix, lưu mô hình ra `models/model.pkl`, lưu metric ra `outputs/metrics.json`, và log thí nghiệm bằng MLflow. Em cũng chạy nhiều thí nghiệm với các thuật toán khác nhau như Logistic Regression, Gradient Boosting, MLP và Random Forest để so sánh hiệu quả. Kết quả tốt nhất hiện tại là Random Forest với accuracy khoảng `0.7500` và weighted F1 khoảng `0.7491`.

Ở bước CI/CD, em dùng DVC để version dữ liệu và cấu hình remote trên Amazon S3. File dữ liệu lớn không được commit trực tiếp vào Git, thay vào đó Git chỉ lưu file `.dvc` chứa hash của dữ liệu. Pipeline GitHub Actions gồm các job `Unit Test`, `Train`, `Eval` và `Deploy`. Job `Eval` kiểm tra ngưỡng accuracy tối thiểu `0.70` và so sánh với production model hiện tại để tránh deploy model yếu hơn. Khi pipeline pass, model được upload lên S3 và service FastAPI trên EC2 được restart để phục vụ endpoint `/health` và `/predict`.

Ở bước continuous training, em bổ sung dữ liệu phase 2 vào tập train, cập nhật DVC pointer và push lên GitHub. Commit dữ liệu mới tự động kích hoạt pipeline train lại, đánh giá lại và deploy lại model mà không cần thao tác thủ công. Sau khi thêm dữ liệu, số mẫu train tăng từ `2998` lên `5996`, DVC md5 của data đổi từ `c43afab731fd6431a94f888fdc687876` sang `5853e7711c78f02286e65fca6cb6e124`.

Em cũng bổ sung cơ chế quản lý version model rõ ràng hơn. Production model nằm ở `s3://teswy-2a202600084-day21-mlops/models/latest/`, còn mỗi model đã deploy được lưu bất biến theo commit SHA tại `models/runs/<git_sha>/`. File `metrics.json` của mỗi model version ghi lại `git_sha`, `github_run_id`, `train_rows`, `data_dvc_md5`, accuracy và F1. Nhờ vậy có thể biết model nào được train từ data version nào, metric ra sao, và có thể rollback/promote lại một model cũ bằng script `scripts/promote_model_version.py`.

## Bộ siêu tham số đã chọn

Em chọn Random Forest làm production model vì kết quả tốt nhất trong các thí nghiệm so sánh:

```yaml
model_type: random_forest
n_estimators: 1200
max_depth: null
min_samples_split: 2
min_samples_leaf: 1
max_features: 0.8
class_weight: balanced_subsample
n_jobs: -1
```

Lý do chọn bộ này là Random Forest mạnh hơn các mô hình thử nghiệm khác trên tập eval cố định. Logistic Regression có kết quả thấp hơn rõ rệt, MLP và Gradient Boosting tốt hơn Logistic Regression nhưng vẫn thấp hơn Random Forest. Việc giữ `params.yaml` ở cấu hình tốt nhất giúp pipeline production không deploy nhầm model yếu.

## Khó khăn và cách xử lý

Khó khăn chính là làm rõ sự khác nhau giữa data version, experiment version và model version. Ban đầu model chỉ ghi đè vào `models/latest`, nên khó chứng minh khả năng chọn lại model cũ. Em đã xử lý bằng cách lưu thêm model vào `models/runs/<git_sha>/`, ghi DVC hash vào `metrics.json`, và thêm script promote/rollback. Một khó khăn khác là các run giống nhau nếu dùng cùng data, cùng hyperparameter và cùng random seed. Em đã bổ sung script `scripts/run_model_comparison.py` để chạy nhiều config khác nhau và tạo bảng so sánh metric rõ ràng.

## Bonus claim

Em có claim đủ 5 bonus:

- **Bonus 1 - DagsHub MLflow tracking:** workflow đã hỗ trợ tracking MLflow từ xa qua DagsHub bằng GitHub Secrets.
- **Bonus 2 - Nhiều thuật toán:** hỗ trợ Logistic Regression, Random Forest, Extra Trees, Gradient Boosting, MLP và optional LNN; đã có bảng so sánh nhiều model.
- **Bonus 3 - Báo cáo hiệu suất tự động:** pipeline upload `outputs/report.txt`, `outputs/metrics.json` và model artifact.
- **Bonus 4 - Rollback guard:** Eval job chặn deploy nếu accuracy mới thấp hơn production model hiện tại; thêm script promote/rollback model version.
- **Bonus 5 - Cảnh báo lệch dữ liệu:** `metrics.json` ghi label distribution và `drift_warnings`; nếu lớp nào dưới 10% sẽ in warning.

## Screenshots/Evidence nên nộp

Các screenshot chính đã được gom vào thư mục `outputs/submission_evidence/` với tên thống nhất:

1. `outputs/submission_evidence/01_dagshub_experiments.png` - DagsHub Experiments có nhiều run và các cột metric như accuracy, f1_score.
2. `outputs/submission_evidence/02_model_comparison_metrics.png` - bảng so sánh nhiều model/thuật toán khác nhau.
3. `outputs/submission_evidence/03_github_actions_eval_gate_failed.png` - run đầu tiên bị chặn ở Eval gate khi accuracy dưới ngưỡng.
4. `outputs/submission_evidence/04_github_actions_data_commit_green.png` - run do commit dữ liệu phase 2 kích hoạt, xanh đủ Unit Test, Train, Eval, Deploy.
5. `outputs/submission_evidence/05_s3_dvc_data_objects.png` - S3 hiển thị dữ liệu đã được DVC push.
6. `outputs/submission_evidence/06_s3_model_versions.png` - S3 hiển thị model version dưới `models/latest/` và `models/runs/<git_sha>/`.
7. `outputs/submission_evidence/07_ec2_instance_running.png` - EC2 instance dùng để phục vụ FastAPI khi bài lab còn chạy.
8. `outputs/submission_evidence/08_api_health_predict.png` - kết quả endpoint `/health` và `/predict`.

Ngoài các screenshot trên, thư mục `outputs/evidence/` chứa thêm evidence dạng JSON/Markdown/TXT để đối chiếu lại khi cần.
