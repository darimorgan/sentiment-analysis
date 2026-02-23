"""Feature extraction from fine-tuned BERT for SVC training."""


import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .model import StableBertClassifier


class FeatureExtractor:
    """Extract hidden features from fine-tuned BERT for SVC."""

    def __init__(self, config: Config):
        self.config = config

    def extract(
        self,
        model: StableBertClassifier,
        data_loader: DataLoader,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Extract hidden features from BERT model.

        Args:
            model: Fine-tuned BERT classifier.
            data_loader: Data loader with samples.

        Returns:
            Tuple of (features, labels). Labels is None if not in dataset.
        """
        model.eval()
        model = model.to(self.config.device)

        all_features = []
        all_labels = []

        # Check if labels exist in dataset
        sample = data_loader.dataset[0]
        has_labels = "labels" in sample

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Extracting features"):
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)

                _, _, hidden_output = model(input_ids, attention_mask)
                all_features.append(hidden_output.cpu().numpy())

                if has_labels:
                    all_labels.append(batch["labels"].numpy())

        features = np.vstack(all_features)

        if has_labels:
            labels = np.concatenate(all_labels)
            return features, labels

        return features, None
