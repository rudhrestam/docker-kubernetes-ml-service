"""Smoke / load test for the ML service.

Hits /health once, then sends N prediction requests (a mix of low-risk and
high-risk synthetic customers), reporting latency stats and a couple of
sample responses. Usage: python smoke_test.py [base_url] [n_requests]
"""
import statistics
import sys
import time

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60

LOW_RISK = {"tenure_months": 48, "monthly_visits": 12, "avg_spend": 55.0,
            "app_sessions_30d": 20, "support_tickets_90d": 0, "sentiment": 0.8}
HIGH_RISK = {"tenure_months": 3, "monthly_visits": 1, "avg_spend": 12.0,
             "app_sessions_30d": 0, "support_tickets_90d": 6, "sentiment": -0.7}


def main() -> None:
    # trust_env=False: a local smoke test must never route through proxies
    # picked up from HTTP_PROXY-style environment variables.
    with httpx.Client(base_url=BASE, timeout=10, trust_env=False) as client:
        h = client.get("/health")
        print(f"GET /health -> {h.status_code} {h.json()}")

        lat = []
        low_p, high_p = None, None
        for i in range(N):
            payload = LOW_RISK if i % 2 == 0 else HIGH_RISK
            t0 = time.perf_counter()
            r = client.post("/predict", json=payload)
            lat.append((time.perf_counter() - t0) * 1000)
            r.raise_for_status()
            if i == 0:
                low_p = r.json()
            if i == 1:
                high_p = r.json()

        lat.sort()
        p95 = lat[int(0.95 * len(lat)) - 1]
        print(f"POST /predict x{N}: mean {statistics.mean(lat):.1f} ms, "
              f"p50 {lat[len(lat)//2]:.1f} ms, p95 {p95:.1f} ms, max {lat[-1]:.1f} ms")
        print(f"low-risk sample  -> {low_p}")
        print(f"high-risk sample -> {high_p}")


if __name__ == "__main__":
    main()
