import os
from pathlib import Path

import boto3
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

S3_BUCKET = os.environ["S3_BUCKET"]
S3_MODEL_KEY = os.environ.get("S3_MODEL_KEY", "models/latest/model.pkl")
MODEL_PATH = Path(os.path.expanduser("~/models/model.pkl"))


def download_model():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    s3.download_file(S3_BUCKET, S3_MODEL_KEY, str(MODEL_PATH))
    print(f"Downloaded s3://{S3_BUCKET}/{S3_MODEL_KEY} to {MODEL_PATH}")


download_model()
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    return {"status": "ok", "model_key": S3_MODEL_KEY}


@app.post("/predict")
def predict(req: PredictRequest):
    if len(req.features) != 12:
        raise HTTPException(
            status_code=400,
            detail="Expected 12 features (wine quality)",
        )

    prediction = int(model.predict([req.features])[0])
    labels = {0: "thap", 1: "trung_binh", 2: "cao"}
    return {"prediction": prediction, "label": labels.get(prediction, "unknown")}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
