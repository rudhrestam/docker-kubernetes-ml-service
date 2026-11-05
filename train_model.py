"""Train a tiny churn-risk model on synthetic data and save it for serving.

Features mirror a small customer-360 style feature set (tenure, usage,
engagement, support load). Fixed seed = reproducible metrics.
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(42)

N = 4000
tenure_months = rng.integers(1, 72, N)
monthly_visits = rng.poisson(6, N)
avg_spend = np.round(rng.gamma(3.0, 12.0, N), 2)
app_sessions_30d = rng.poisson(9, N)
support_tickets_90d = rng.poisson(1.2, N)
sentiment = np.round(rng.uniform(-1, 1, N), 3)

# Synthetic ground truth: churn risk rises with support load and low
# engagement/sentiment, falls with tenure and visits.
logit = (0.55 * support_tickets_90d - 0.16 * monthly_visits
         - 0.05 * tenure_months - 1.1 * sentiment
         - 0.06 * app_sessions_30d + 1.4)
p = 1 / (1 + np.exp(-logit))
y = (rng.uniform(0, 1, N) < p).astype(int)

X = np.column_stack([tenure_months, monthly_visits, avg_spend,
                     app_sessions_30d, support_tickets_90d, sentiment])
FEATURES = ["tenure_months", "monthly_visits", "avg_spend",
            "app_sessions_30d", "support_tickets_90d", "sentiment"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_tr, y_tr)

pred = model.predict(X_te)
proba = model.predict_proba(X_te)[:, 1]
metrics = {
    "train_rows": int(len(X_tr)),
    "test_rows": int(len(X_te)),
    "test_accuracy": round(float(accuracy_score(y_te, pred)), 4),
    "test_roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
    "churn_rate_test": round(float(y_te.mean()), 4),
    "features": FEATURES,
}
joblib.dump(model, HERE / "model.joblib")
(HERE / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
