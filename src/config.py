"""Centralized, env-var-driven configuration."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # inference
    model_dir: str = "model_artifacts"
    model_source: str = "local"          # "local" | "hf_hub"
    hf_model_repo: str = ""
    max_seq_length: int = 128
    port: int = 8080

    # training defaults (overridable via CLI flags in src/train.py)
    base_model_name: str = "distilbert-base-uncased"
    default_batch_size: int = 32
    default_learning_rate: float = 5e-5
    default_num_epochs: int = 5
    default_layers_to_train: int = 4
    train_split_percentage: float = 0.8
    random_seed: int = 42


settings = Settings()
