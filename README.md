# AI Request Dispatcher — Production Service

A production build of the "AI Powered Request Dispatcher": a fine-tuned DistilBERT
classifier that routes a user instruction to one of five intent categories
(`q_and_a`, `classification`, `information_distillation`, `brainstorming`,
`creative_writing`), served behind a FastAPI HTTP API and packaged as a Docker
image you can deploy anywhere.

```
ai-dispatcher/
├── src/                 # training pipeline (data, model, train, evaluate)
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── app/                 # inference API
│   ├── main.py
│   └── schemas.py
├── tests/                # pytest unit + API tests
├── deploy/               # ready-made configs for common platforms
├── Dockerfile
├── docker-compose.yml
├── requirements.txt         
└── .github/workflows/ci.yml
```

## 1. Local setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # full dev/train/eval/serve environment
cp .env.example .env
```

`requirements.txt` pulls in `requirements-serve.txt` (the minimal set the API
needs) plus training/eval-only deps (`pandas`, `scikit-learn`, `torchmetrics`,
`matplotlib`, `seaborn`, `pytest`). The Docker image installs only
`requirements-serve.txt`, keeping the served container lean.

Put your dataset at `data/databricks-dolly_augmented.csv` (needs `instruction`
and `category` columns — this is the same file used in the original notebook).

## 2. Train

```bash
python -m src.train \
  --data-path data/databricks-dolly_augmented.csv \
  --output-dir model_artifacts \
  --layers-to-train 4 \
  --learning-rate 5e-5 \
  --num-epochs 5 \
  --batch-size 32
```

This mirrors the notebook's `helper_utils.py` logic exactly (same
`download_bert`/`load_bert` two-step, `AdamW(weight_decay=0.01)`,
macro-averaged F1 and confusion matrix via `torchmetrics`, 128-token
truncation), wrapped as a repeatable CLI job. It will:

1. Load + clean the CSV, merge granular categories (`general_qa`/`open_qa`/`closed_qa` → `q_and_a`,
   `information_extraction`/`summarization` → `information_distillation`).
2. Download `distilbert-base-uncased` once to `--base-model-cache-dir`
   (`./distilbert-local-base` by default) and reuse it on subsequent runs —
   no Hub hit once cached, so this also works in air-gapped CI.
3. Fine-tune with class-weighted loss and partial layer freezing (only the
   last N transformer layers + classification head are trained).
4. Track train/val loss, accuracy, and macro-F1 each epoch; compute a
   confusion matrix over the final epoch's validation set.
5. Save to `model_artifacts/`: the HF model + tokenizer, `id2cat.json`
   (label→category mapping), and `training_metrics.json` (metrics + full
   per-epoch history + confusion matrix).

Target from the original assignment: validation loss < 0.7, validation F1 ≥ 0.8.

Generate a report (per-class accuracy table, confusion matrix PNG, and a
qualitative check against held-out prompts) after training:

```bash
python -m src.evaluate --model-dir model_artifacts
```

Run the unit tests against the pipeline code (no GPU/model download required
for most of them):

```bash
pytest tests/ -v
```

## 3. Serve locally

```bash
export MODEL_DIR=model_artifacts
uvicorn app.main:app --reload --port 8080
```

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Summarize the plot of Hamlet in two sentences."}'
```

```json
{
  "category": "information_distillation",
  "confidence": 0.94,
  "probabilities": {
    "q_and_a": 0.02,
    "classification": 0.01,
    "information_distillation": 0.94,
    "brainstorming": 0.01,
    "creative_writing": 0.02
  }
}
```

`GET /health` returns `{"status": "ok", "model_loaded": true}` — use it as your
container health check / readiness probe.

## 4. Docker

```bash
docker build -t ai-dispatcher:latest .
docker run -p 8080:8080 -v $(pwd)/model_artifacts:/app/model_artifacts ai-dispatcher:latest
```

Or with `docker-compose up` (uses the same volume mount).

The image doesn't bake the trained weights in by default — it mounts/reads
`model_artifacts/` at runtime via `MODEL_DIR`. For a self-contained image (e.g.
for platforms that don't support volumes, like most PaaS/serverless targets),
either:
- `COPY model_artifacts/ ./model_artifacts/` in the Dockerfile before building, or
- push the trained model to the Hugging Face Hub and set `MODEL_SOURCE=hf_hub`
  + `HF_MODEL_REPO=your-username/ai-dispatcher` (see `app/main.py`) so the
  container pulls weights at startup instead of needing them on disk.

## 5. Deploying

Any platform that runs a container will work. Ready-made starting points are
in `deploy/`:

| Platform | File | Notes |
|---|---|---|
| Google Cloud Run | `deploy/deploy_cloudrun.sh` | serverless, scales to zero, easiest for a demo API |
| Fly.io | `deploy/fly.toml` | `fly launch`, cheap always-on small VM |
| Render | `deploy/render.yaml` | one-click "Blueprint" deploy from repo |
| Hugging Face Spaces | `deploy/README_hf_space.md` | free hosting, good if the model itself lives on the HF Hub |

Pick one once you know your constraints (budget, always-on vs. scale-to-zero,
GPU vs. CPU inference — this model is small enough to run fine on CPU).

`.github/workflows/ci.yml` runs `pytest` on every push and, on pushes to
`main`, builds and pushes the Docker image to GitHub Container Registry
(`ghcr.io/<owner>/<repo>`) — point whichever platform you pick at that image.

## 6. Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_DIR` | `model_artifacts` | local path to load model+tokenizer+id2cat from |
| `MODEL_SOURCE` | `local` | `local` or `hf_hub` |
| `HF_MODEL_REPO` | — | HF Hub repo id, used when `MODEL_SOURCE=hf_hub` |
| `MAX_SEQ_LENGTH` | `128` | tokenizer truncation length at inference |
| `PORT` | `8080` | uvicorn port (matches Dockerfile `EXPOSE`) |
