"""Dataset class for sentiment analysis."""

from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class RatingDataset(Dataset):
    """PyTorch Dataset for rating/sentiment classification."""

    def __init__(
        self,
        texts: list[str],
        targets: Optional[list[int]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        """
        Initialize the dataset.

        Args:
            texts: List of text samples.
            targets: List of target labels (0-indexed). None for inference.
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum sequence length.
        """
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

        if self.targets is not None:
            item["labels"] = torch.tensor(self.targets[idx], dtype=torch.long)

        return item