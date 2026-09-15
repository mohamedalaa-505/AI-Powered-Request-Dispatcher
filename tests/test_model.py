from transformers import DistilBertConfig, DistilBertForSequenceClassification

from src.model import NUM_TRANSFORMER_LAYERS, partially_freeze_bert_layers


def _tiny_model(num_labels=5):
    """Builds a randomly-initialized, tiny DistilBERT so tests don't need network access."""
    config = DistilBertConfig(
        vocab_size=100,
        dim=16,
        hidden_dim=32,
        n_layers=NUM_TRANSFORMER_LAYERS,
        n_heads=2,
        num_labels=num_labels,
    )
    return DistilBertForSequenceClassification(config)


def test_partial_freeze_freezes_early_layers_and_trains_late_ones():
    model = _tiny_model()
    partially_freeze_bert_layers(model, layers_to_train=2)

    early = dict(model.named_parameters())["distilbert.transformer.layer.0.attention.q_lin.weight"]
    late = dict(model.named_parameters())["distilbert.transformer.layer.5.attention.q_lin.weight"]
    classifier = dict(model.named_parameters())["classifier.weight"]

    assert early.requires_grad is False
    assert late.requires_grad is True
    assert classifier.requires_grad is True


def test_partial_freeze_clamps_out_of_range_values():
    model = _tiny_model()
    # Negative should clamp to 0 trainable transformer layers.
    partially_freeze_bert_layers(model, layers_to_train=-5)
    layer0 = dict(model.named_parameters())["distilbert.transformer.layer.0.attention.q_lin.weight"]
    assert layer0.requires_grad is False
    # Classifier head is always trainable regardless.
    assert dict(model.named_parameters())["classifier.weight"].requires_grad is True

    model2 = _tiny_model()
    # Too-large should clamp to NUM_TRANSFORMER_LAYERS (all trainable).
    partially_freeze_bert_layers(model2, layers_to_train=999)
    layer0_v2 = dict(model2.named_parameters())["distilbert.transformer.layer.0.attention.q_lin.weight"]
    assert layer0_v2.requires_grad is True
