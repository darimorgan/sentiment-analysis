"""Inference pipeline for sentiment prediction."""

from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.stats import mode
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .config import Config
from .dataset import RatingDataset
from .features import FeatureExtractor
from .model import StableBertClassifier


class SentimentPredictor:
    """Ensemble predictor using fine-tuned BERT (direct or + SVC/LogReg)."""

    def __init__(
        self,
        config: Config,
        model_dir: str | Path | None = None,
        classifier: str = "direct",
    ):
        self.config = config
        self.model_dir = Path(model_dir) if model_dir else config.model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.classifier = classifier

        if classifier != "direct":
            self.feature_extractor = FeatureExtractor(config)

        self.bert_models: list[StableBertClassifier] = []
        self.svc_models: list = []

    def load_models(self, num_folds: int = 5) -> None:
        """Load all fold models for ensemble prediction."""
        print(f"Loading {num_folds} fold models...")

        for fold_idx in range(num_folds):
            # Load BERT
            bert_path = self.model_dir / f"fold_{fold_idx}_best_bert.pt"
            bert_model = StableBertClassifier(
                model_name=self.config.model_name,
                num_classes=self.config.num_classes,
                dropout=self.config.dropout_rate,
                hidden_dim=self.config.hidden_dim,
            )
            bert_model.load_state_dict(
                torch.load(bert_path, map_location=self.config.device)
            )
            bert_model.to(self.config.device)
            bert_model.eval()
            self.bert_models.append(bert_model)

            # Load classifier (skip for direct mode)
            if self.classifier != "direct":
                clf_path = self.model_dir / f"fold_{fold_idx}_clf.joblib"
                clf_model = joblib.load(clf_path)
                self.svc_models.append(clf_model)

        print(f"Loaded {len(self.bert_models)} BERT models")

    def predict(
        self,
        texts: list[str],
        batch_size: int = 8,
        return_all_predictions: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Predict sentiment for texts using ensemble voting.

        Args:
            texts: List of text samples.
            batch_size: Batch size for inference.
            return_all_predictions: If True, also return individual fold predictions.

        Returns:
            Ensemble predictions (1-5 scale).
            If return_all_predictions=True, also returns array of all fold predictions.
        """
        if not self.bert_models:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        dataset = RatingDataset(
            texts=texts,
            targets=None,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
        )
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_fold_predictions = []

        for fold_idx, bert_model in enumerate(self.bert_models):
            print(f"Predicting with fold {fold_idx + 1}...")

            if self.classifier == "direct":
                # Predict using BERT logits directly
                fold_preds = []
                with torch.no_grad():
                    for batch in data_loader:
                        input_ids = batch["input_ids"].to(self.config.device)
                        attention_mask = batch["attention_mask"].to(self.config.device)
                        _, logits, _ = bert_model(input_ids, attention_mask)
                        preds = torch.argmax(logits, dim=1).cpu().numpy()
                        fold_preds.extend(preds)
                fold_preds = np.array(fold_preds)
            else:
                # Extract features + classifier prediction
                features, _ = self.feature_extractor.extract(bert_model, data_loader)
                fold_preds = self.svc_models[fold_idx].predict(features)

            all_fold_predictions.append(fold_preds)

        # Ensemble with majority voting
        predictions_stack = np.stack(all_fold_predictions)
        ensemble_preds_0_indexed = mode(predictions_stack, axis=0)[0].flatten()

        # Convert to 1-5 scale
        ensemble_preds = ensemble_preds_0_indexed + 1

        if return_all_predictions:
            return ensemble_preds, predictions_stack + 1

        return ensemble_preds

    def predict_single(self, text: str) -> int:
        """Predict sentiment for a single text."""
        predictions = self.predict([text])
        return int(predictions[0])
