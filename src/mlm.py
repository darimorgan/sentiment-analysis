"""Domain-adaptive MLM pretraining for BERT."""

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

from .config import Config


class MLMDataset(Dataset):
    """Dataset for masked language modeling."""

    def __init__(self, texts: list[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_special_tokens_mask=True,
        )
        return {k: torch.tensor(v) for k, v in encoding.items()}


def mlm_pretrain(texts: list[str], config: Config) -> str:
    """
    Run MLM pretraining on domain texts and save adapted model.

    Args:
        texts: All available texts for unsupervised pretraining.
        config: Training configuration.

    Returns:
        Path to the saved adapted model.
    """
    print(f"\n{'='*60}")
    print("MLM DOMAIN PRETRAINING")
    print(f"{'='*60}")
    print(f"Texts: {len(texts)}, Epochs: {config.mlm_epochs}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForMaskedLM.from_pretrained(config.model_name)
    model = model.to(config.device)

    dataset = MLMDataset(texts, tokenizer, config.max_length)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=config.mlm_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.mlm_learning_rate)

    num_training_steps = len(data_loader) * config.mlm_epochs
    num_warmup_steps = int(num_training_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    scaler = torch.amp.GradScaler("cuda")
    accumulation_steps = 4

    model.train()
    for epoch in range(config.mlm_epochs):
        total_loss = 0
        optimizer.zero_grad()
        progress = tqdm(data_loader, desc=f"MLM Epoch {epoch + 1}/{config.mlm_epochs}")

        for step, batch in enumerate(progress):
            batch = {k: v.to(config.device) for k, v in batch.items()}

            with torch.amp.autocast("cuda"):
                outputs = model(**batch)
                loss = outputs.loss / accumulation_steps

            scaler.scale(loss).backward()
            total_loss += loss.item() * accumulation_steps

            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            progress.set_postfix(loss=f"{loss.item() * accumulation_steps:.4f}")

        avg_loss = total_loss / len(data_loader)
        print(f"MLM Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}")

    # Save adapted model
    output_path = str(config.model_dir / "mlm_pretrained")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"MLM pretrained model saved to {output_path}")
    return output_path
