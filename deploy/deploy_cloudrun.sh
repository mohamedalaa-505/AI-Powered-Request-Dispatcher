#!/usr/bin/env bash
# Deploy the AI Request Dispatcher to Google Cloud Run.
#
# Prereqs:
#   - gcloud CLI installed and authenticated (`gcloud auth login`)
#   - a GCP project with billing + Cloud Run + Artifact Registry APIs enabled
#   - model weights either baked into the image (uncomment the COPY line in
#     the Dockerfile) or MODEL_SOURCE=hf_hub / HF_MODEL_REPO set below
#
# Usage:
#   PROJECT_ID=my-gcp-project ./deploy/deploy_cloudrun.sh

set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID env var to your GCP project id}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-ai-dispatcher}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${SERVICE_NAME}/${SERVICE_NAME}:latest"

gcloud config set project "${PROJECT_ID}"

gcloud artifacts repositories create "${SERVICE_NAME}" \
  --repository-format=docker --location="${REGION}" \
  --description="AI Request Dispatcher" 2>/dev/null || true

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

docker build -t "${IMAGE}" .
docker push "${IMAGE}"

gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 3 \
  --set-env-vars "MODEL_SOURCE=${MODEL_SOURCE:-local},HF_MODEL_REPO=${HF_MODEL_REPO:-}"

echo "Deployed. Fetching URL..."
gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format 'value(status.url)'
