# Deploying to Hugging Face Spaces (Docker SDK)

1. Push your trained model to the Hub:

   ```python
   from huggingface_hub import HfApi
   HfApi().create_repo("your-username/ai-dispatcher-model", exist_ok=True)
   trained_model.push_to_hub("your-username/ai-dispatcher-model")
   tokenizer.push_to_hub("your-username/ai-dispatcher-model")
   ```

2. Create a new Space at https://huggingface.co/new-space, SDK = **Docker**.

3. Push this repo's contents to the Space's git remote (Spaces are just git
   repos). The existing `Dockerfile` works as-is — Spaces route external
   traffic to port 7860 by default, so either:
   - add `EXPOSE 7860` and run uvicorn on `--port 7860`, or
   - set the Space's "App port" setting to `8080` to match this Dockerfile.

4. In the Space's Settings > Variables, set:
   - `MODEL_SOURCE=hf_hub`
   - `HF_MODEL_REPO=your-username/ai-dispatcher-model`

The container will then pull the model from the Hub at startup instead of
needing local `model_artifacts/`.
