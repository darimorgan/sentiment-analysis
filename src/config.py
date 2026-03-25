"""Configuration and hyperparameters for sentiment analysis."""

from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class Config:
    """Central configuration for training and inference."""

    # Model
    model_name: str = "DeepPavlov/rubert-base-cased-conversational"
    num_classes: int = 5
    max_length: int = 512
    dropout_rate: float = 0.3
    hidden_dim: int = 384

    # Training
    batch_size: int = 8
    accumulation_steps: int = 4
    num_epochs: int = 5
    num_folds: int = 5
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    layer_decay: float = 0.95

    # ArcFace
    arcface_margin: float = 0.5
    arcface_scale: float = 64.0
    label_smoothing: float = 0.0

    # MLM pretraining
    mlm_epochs: int = 2
    mlm_learning_rate: float = 2e-5
    mlm_batch_size: int = 4

    # SVC
    svc_c: float = 1.0
    svc_kernel: str = "rbf"
    svc_cache_size: int = 8000

    # LogReg
    logreg_c: float = 1.0
    logreg_penalty: str = "l2"
    logreg_max_iter: int = 2000

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    model_dir: Path = field(default_factory=lambda: Path("models"))

    # Data columns
    text_column: str = "text"
    label_column: str = "rate"

    # Device
    device: str = "auto"

    def __post_init__(self):
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.accumulation_steps
