# Docker / Kubernetes ML Service (Demo)

![CI](https://github.com/rudhrestam/docker-kubernetes-ml-service/actions/workflows/ci.yml/badge.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A small ML inference service, packaged the way a production one is: a scikit-learn model trained on synthetic data, served by FastAPI, containerised with a Dockerfile and docker-compose, and deployed with Kubernetes manifests that carry health probes and resource limits.

> ⚠️ **Synthetic data only.** The model trains on generated customer-style features with a fixed seed. No real customer data — anyone's — is used.

> This is a representative demo of patterns used in production systems, built with original code. It contains no employer code, data, or internal details.

## Layout

```
train_model.py     trains a RandomForest churn-risk model, writes model.joblib + model_metrics.json
app.py             FastAPI service: GET /health, POST /predict
smoke_test.py      hits /health, then N predictions, reports latency + sample outputs
Dockerfile         python:3.12-slim; trains the model at image build; HEALTHCHECK on /health
docker-compose.yml one-command local run on port 8000
k8s/deployment.yaml  2 replicas, resources requests/limits, liveness + readiness probes on /health
k8s/service.yaml     ClusterIP service, port 80 -> 8000
```

## Real run output

Model training (4,000 synthetic customers, 75/25 split):

```
train_rows: 3000   test_rows: 1000
test_accuracy: 0.77   test_roc_auc: 0.7964   churn_rate_test: 0.304
```

Service run (uvicorn, this sandbox) + smoke test, 60 predictions:

```
GET /health -> 200 {'status': 'ok', 'model_loaded': True, 'test_roc_auc': 0.7964}
POST /predict x60: mean 49.0 ms, p50 39.4 ms, p95 63.9 ms, max 349.5 ms
low-risk sample  -> {'churn_probability': 0.05,  'churn_label': 0}
high-risk sample -> {'churn_probability': 0.885, 'churn_label': 1}
```

The two sample customers are deliberately opposite profiles (long tenure, no support tickets, positive sentiment vs new, disengaged, six tickets, negative sentiment) — the model separates them cleanly, which is the smoke test's real assertion.

All three YAML files (`k8s/deployment.yaml`, `k8s/service.yaml`, `docker-compose.yml`) parse and validate as YAML, and the probes in the Deployment point at the same `/health` endpoint the smoke test exercises.

> **Docker note, honestly:** the sandbox this was built in has no Docker daemon, so `docker build` itself was not run here. Everything the image would run was verified directly instead: the training script, the FastAPI app under uvicorn, and the smoke test above. The Dockerfile is a standard slim-image build (install requirements → copy code → train → serve) with nothing environment-specific in it.

## Run it

```bash
pip install -r requirements.txt
python train_model.py
uvicorn app:app --port 8000
python smoke_test.py http://127.0.0.1:8000 60
```

Or with containers:

```bash
docker compose up --build        # service on http://localhost:8000
kubectl apply -f k8s/            # deployment + service in a cluster
```

Example request:

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"tenure_months": 3, "monthly_visits": 1, "avg_spend": 12.0,
       "app_sessions_30d": 0, "support_tickets_90d": 6, "sentiment": -0.7}'
```

## Tech stack

Python 3.10+ · FastAPI · scikit-learn (RandomForest) · uvicorn · Docker · docker-compose · Kubernetes

## What it demonstrates

- End-to-end ML serving: train → persist → serve → smoke-test, with held-out metrics shipped alongside the model and surfaced by the API
- Input validation at the API boundary (pydantic ranges on every feature)
- Container packaging: reproducible training inside the image build, Docker HEALTHCHECK
- Kubernetes fundamentals done properly: probes wired to a real health endpoint, resource requests/limits, replica count, label/selector consistency between Deployment and Service
- A smoke test that asserts behaviour (risk separation), not just HTTP 200s
