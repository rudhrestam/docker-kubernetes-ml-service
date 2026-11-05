"""FastAPI inference service for the synthetic churn-risk model.

Endpoints:
  GET  /health   -> liveness/readiness payload (model loaded or not)
  POST /predict  -> churn probability + label for one customer feature set
"""
import json
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.joblib"
METRICS_PATH = HERE / "model_metrics.json"

app = FastAPI(title="churn-risk-ml-service", version="1.0.0")

_model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
_metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}


class CustomerFeatures(BaseModel):
    tenure_months: int = Field(ge=0, le=600)
    monthly_visits: int = Field(ge=0, le=200)
    avg_spend: float = Field(ge=0)
    app_sessions_30d: int = Field(ge=0, le=500)
    support_tickets_90d: int = Field(ge=0, le=100)
    sentiment: float = Field(ge=-1, le=1)


@app.get("/health")
def health():
    return {"status": "ok" if _model is not None else "model_missing",
            "model_loaded": _model is not None,
            "test_roc_auc": _metrics.get("test_roc_auc")}


@app.post("/predict")
def predict(features: CustomerFeatures):
    row = np.array([[features.tenure_months, features.monthly_visits,
                     features.avg_spend, features.app_sessions_30d,
                     features.support_tickets_90d, features.sentiment]])
    proba = float(_model.predict_proba(row)[0, 1])
    return {"churn_probability": round(proba, 4),
            "churn_label": int(proba >= 0.5),
            "model_test_roc_auc": _metrics.get("test_roc_auc")}
