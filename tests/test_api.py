import torch
from fastapi.testclient import TestClient

from app.main import app, state


class FakeOutputs:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    """Deterministic stand-in for a trained classifier: always favors class 2."""

    def __call__(self, **kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        logits = torch.tensor([[0.1, 0.1, 5.0, 0.1, 0.1]] * batch_size)
        return FakeOutputs(logits)


class FakeEncoding(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    def __call__(self, text, truncation=True, max_length=128, return_tensors="pt"):
        return FakeEncoding({
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        })


def _inject_fake_model():
    state["model"] = FakeModel()
    state["tokenizer"] = FakeTokenizer()
    state["device"] = torch.device("cpu")
    state["id2cat"] = {
        0: "q_and_a", 1: "classification", 2: "information_distillation",
        3: "brainstorming", 4: "creative_writing",
    }


def test_health_before_model_load_reports_status():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert "status" in resp.json()


def test_predict_returns_expected_category():
    with TestClient(app) as client:
        _inject_fake_model()
        resp = client.post("/predict", json={"text": "Summarize this document for me."})
        assert resp.status_code == 200
        body = resp.json()
        assert body["category"] == "information_distillation"
        assert 0.0 <= body["confidence"] <= 1.0
        assert set(body["probabilities"].keys()) == set(state["id2cat"].values())


def test_predict_requires_nonempty_text():
    with TestClient(app) as client:
        _inject_fake_model()
        resp = client.post("/predict", json={"text": ""})
        assert resp.status_code == 422


def test_categories_endpoint():
    with TestClient(app) as client:
        _inject_fake_model()
        resp = client.get("/categories")
        assert resp.status_code == 200
        assert set(resp.json()["categories"]) == set(state["id2cat"].values())
