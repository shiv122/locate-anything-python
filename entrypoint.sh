#!/usr/bin/env bash
# Start the FastAPI server. The LocateAnything-3B weights load on startup
# (FastAPI lifespan) and stay resident on the GPU; the first boot waits for that
# load before it accepts connections.
set -euo pipefail

echo "[entrypoint] starting LocateAnything API on :${PORT:-8080} (loading model, first boot may take a minute)..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8080}"
