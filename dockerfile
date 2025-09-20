# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Shanghai

# System deps (for psycopg2 and tzdata)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev tzdata ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the application code
COPY main.py /app/main.py

# Default command runs the daily job; override in scheduler/backfill
CMD ["python", "main.py", "--mode", "daily"]

# Healthcheck: verifies python env and presence of DB_URL env (adjust as needed)
# This does not try to connect to DB to avoid failing when DB is temporarily down.
# HEALTHCHECK --interval=1m --timeout=10s --start-period=20s --retries=3 \
#   CMD python - <<'PY' || exit 1
# import os, sys
# req_env = ["DB_URL"]
# missing = [k for k in req_env if not os.getenv(k)]
# if missing:
#     print(f"Missing env: {missing}", file=sys.stderr); sys.exit(1)
# # quick import check
# try:
#     import akshare, pandas, sqlalchemy, psycopg2
# except Exception as e:
#     print(f"Import error: {e}", file=sys.stderr); sys.exit(1)
# print("OK")
# PY