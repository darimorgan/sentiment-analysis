"""BERT model with ArcFace loss for sentiment classification."""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel


class StableArcFaceLoss(nn.Module):
    """
    ArcFace loss combined with CrossEntropy for better class separation.

    ArcFace adds an angular margin penalty to improve discriminative power
    of the learned embeddings.
    """

    def __init__(
        self,
        embedding_size: int,
        num_classes: int,
        margin: float = 0.5,
        scale: float = 64.0,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.num_classes = num_classes

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined CrossEntropy + ArcFace loss.

        Args:
            embeddings: Hidden layer embeddings (batch_size, embedding_size).
            labels: Ground truth labels (batch_size,).
            logits: Classification logits (batch_size, num_classes).

        Returns:
            Combined loss value.
        """
        normalized_embeddings = nn.functional.normalize(embeddings)
        normalized_weight = nn.functional.normalize(self.weight)

        cosine = torch.matmul(normalized_embeddings, normalized_weight.t())
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        # ArcFace margin calculation
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        margin_tensor = torch.tensor(
            self.margin, device=cosine.device, dtype=cosine.dtype
        )

        phi = cosine * torch.cos(margin_tensor) - sine * torch.sin(margin_tensor)
        arcface_logits = phi * self.scale

        # Combined loss with gradual ArcFace introduction
        ce_loss = self.ce_loss(logits, labels)
        arcface_loss = self.ce_loss(arcface_logits, labels)

        return ce_loss + 0.05 * arcface_loss


class StableBertClassifier(nn.Module):
    """
    BERT-based classifier with ArcFace loss for sentiment analysis.

    Architecture:
        BERT -> Dropout -> Linear(768, 384) -> GELU -> Dropout -> Linear(384, num_classes)
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 5,
        dropout: float = 0.3,
        hidden_dim: int = 384,
        arcface_margin: float = 0.5,
        arcface_scale: float = 64.0,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert_hidden_size = self.bert.config.hidden_size

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.hidden = nn.Linear(self.bert_hidden_size, hidden_dim)
        self.activation = nn.GELU()
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.arcface = StableArcFaceLoss(
            embedding_size=hidden_dim,
            num_classes=num_classes,
            margin=arcface_margin,
            scale=arcface_scale,
            label_smoothing=label_smoothing,
        )

        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs (batch_size, seq_length).
            attention_mask: Attention mask (batch_size, seq_length).
            labels: Optional ground truth labels for loss computation.

        Returns:
            Tuple of (loss, logits, hidden_features).
            Loss is None if labels not provided.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # [CLS] token pooling
        pooled_output = outputs.last_hidden_state[:, 0, :]

        hidden_output = self.activation(self.hidden(self.dropout1(pooled_output)))
        logits = self.classifier(self.dropout2(hidden_output))

        loss = None
        if labels is not None:
            loss = self.arcface(hidden_output, labels, logits)

        return loss, logits, hidden_output

    def get_hidden_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract hidden features for SVC training."""
        _, _, hidden_output = self.forward(input_ids, attention_mask)
        return hidden_output