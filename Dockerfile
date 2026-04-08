# ─────────────────────────────────────────────────────────────────────────────
# DataCleaning OpenEnv — Dockerfile
# Compatible with Hugging Face Spaces (port 7860, USER 1000)
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# HF Spaces runs as non-root UID 1000
ARG UID=1000
ARG GID=1000

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Create non-root user ─────────────────────────────────────────────────────
RUN groupadd -g ${GID} appuser && \
    useradd  -u ${UID} -g appuser -m -s /bin/bash appuser

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies (as root for global install) ──────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy source ───────────────────────────────────────────────────────────────
COPY env/        ./env/
COPY server/     ./server/
COPY openenv.yaml .
COPY inference.py .

# ── Permissions ───────────────────────────────────────────────────────────────
RUN chown -R appuser:appuser /app
USER appuser

# ── Environment ───────────────────────────────────────────────────────────────
ENV HOST=0.0.0.0
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# HF Spaces uses port 7860
EXPOSE 7860

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# ── Entry point ───────────────────────────────────────────────────────────────
CMD ["python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--log-level", "info"]
