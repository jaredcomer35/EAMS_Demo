# syntax=docker/dockerfile:1
FROM python:3.11-slim

# (Optional) curl for healthcheck; remove if you don't want it
RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m appuser
WORKDIR /app

# Install deps first (cache-friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + assets
COPY app.py ./
COPY assets/ ./assets/
# If you have a Streamlit config, copy it too:
# COPY .streamlit/ ./.streamlit/

#Add Changelog
COPY CHANGELOG.md ./
# Create a writable data dir for profile persistence
RUN mkdir -p /data && chown -R appuser:appuser /data

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://localhost:8501/healthz || exit 1

USER appuser
# Persisted profiles default here unless overridden via env
ENV PROFILE_STORE_PATH=/data/profiles.json \
    MYSQL_PROFILE_STORE_PATH=/data/mysql_profiles.json

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
