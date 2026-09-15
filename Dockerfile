FROM python:3.11-slim

WORKDIR /app

# System deps for torch/transformers wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-serve.txt

COPY app/ ./app/
COPY src/ ./src/

# Trained weights are expected to be mounted at runtime via `-v $(pwd)/model_artifacts:/app/model_artifacts`.
# To bake weights into the image instead (e.g. for platforms without volume
# support), uncomment the line below after training:
# COPY model_artifacts/ ./model_artifacts/

ENV PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/model_artifacts \
    PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
