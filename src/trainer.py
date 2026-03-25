"""Training utilities for BERT fine-tuning."""

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from .config import Config
from .model import StableBertClassifier


class BertTrainer:
    """Trainer for fine-tuning BERT with ArcFace loss."""

    def __init__(self, config: Config):
        self.config = config

    def get_optimizer_with_llrd(self, model: StableBertClassifier) -> AdamW:
        """
        Create optimizer with Layer-wise Learning Rate Decay (LLRD).

        Lower layers get smaller learning rates, higher layers get larger ones.
        This helps preserve pretrained knowledge in lower layers.
        """
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        optimizer_grouped_parameters = []

        bert_layers = [model.bert.embeddings] + list(model.bert.encoder.layer)
        num_layers = len(bert_layers)

        for layer_idx, layer in enumerate(bert_layers):
            layer_lr = self.config.learning_rate * (
                self.config.layer_decay ** (num_layers - layer_idx)
            )

            # Parameters with weight decay
            params_decay = {
                "params": [
                    p
                    for n, p in layer.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
                "lr": layer_lr,
            }

            # Parameters without weight decay
            params_no_decay = {
                "params": [
                    p
                    for n, p in layer.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": layer_lr,
            }

            optimizer_grouped_parameters.extend([params_decay, params_no_decay])

        # Classifier and hidden layers (higher LR)
        classifier_params = {
            "params": list(model.classifier.parameters())
            + list(model.hidden.parameters()),
            "weight_decay": self.config.weight_decay,
            "lr": self.config.learning_rate * 2.0,
        }

        arcface_params = {
            "params": model.arcface.parameters(),
            "weight_decay": self.config.weight_decay,
            "lr": self.config.learning_rate * 1.5,
        }

        optimizer_grouped_parameters.extend([classifier_params, arcface_params])

        return AdamW(optimizer_grouped_parameters, eps=1e-8)

    def train_fold(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model: StableBertClassifier,
        fold_idx: int,
    ) -> StableBertClassifier:
        """
        Train model for one fold.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            model: BERT classifier model.
            fold_idx: Current fold index.

        Returns:
            Trained model with best validation F1.
        """
        print(f"\n=== Training Fold {fold_idx + 1} ===")

        model = model.to(self.config.device)
        optimizer = self.get_optimizer_with_llrd(model)
        scaler = torch.amp.GradScaler(self.config.device)

        num_training_steps = (
            len(train_loader) * self.config.num_epochs // self.config.accumulation_steps
        )
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        best_val_f1 = 0.0
        best_model_path = self.config.model_dir / f"fold_{fold_idx}_best_bert.pt"

        for epoch in range(self.config.num_epochs):
            # Training
            model.train()
            total_train_loss = 0
            optimizer.zero_grad()

            progress_bar = tqdm(
                train_loader, desc=f"Fold {fold_idx + 1} Epoch {epoch + 1}"
            )

            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                labels = batch["labels"].to(self.config.device)

                with torch.amp.autocast(self.config.device):
                    loss, _, _ = model(input_ids, attention_mask, labels)
                    loss = loss / self.config.accumulation_steps

                scaler.scale(loss).backward()

                loss_val = loss.item() * self.config.accumulation_steps
                if not np.isnan(loss_val):
                    total_train_loss += loss_val

                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.max_grad_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

            # Validation
            val_f1 = self._evaluate(model, val_loader)
            avg_train_loss = total_train_loss / len(train_loader)

            print(f"\nFold {fold_idx + 1}, Epoch {epoch + 1}:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val F1: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), best_model_path)
                print(f"  New best model saved with F1: {best_val_f1:.4f}")

        # Load best model
        model.load_state_dict(torch.load(best_model_path))
        return model

    def _evaluate(self, model: StableBertClassifier, data_loader: DataLoader) -> float:
        """Evaluate model and return weighted F1 score."""
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                labels = batch["labels"].to(self.config.device)

                _, logits, _ = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return f1_score(all_labels, all_preds, average="weighted")
