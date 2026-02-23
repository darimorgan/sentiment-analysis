# Russian Sentiment Analysis

Fine-tuned BERT with ArcFace loss + SVM ensemble for 5-class sentiment classification on Russian text reviews.

## Notebooks

| Notebook | Description | Link |
|----------|-------------|------|
| Fine-tuned BERT + SVC | Main solution with ArcFace loss and ensemble | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/darimorgan/sentiment-analysis/blob/main/notebooks/bert_sentiment.ipynb) |
| Baselines | Logistic Regression & CatBoost with frozen BERT | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/darimorgan/sentiment-analysis/blob/main/notebooks/logistic_regression_catboost_baseline.ipynb) |

## Results

| Model | Test F1    |
|-------|------------|
| Frozen BERT + Logistic Regression | 0.6200     |
| Frozen BERT + CatBoost | ~0.5972    |
| **Fine-tuned BERT + SVC Ensemble** | **0.6599** |

## Approach

1. **Fine-tune BERT** (default: `DeepPavlov/rubert-base-cased-conversational`, configurable via `--model-name`) with:
   - ArcFace loss for better class separation
   - Layer-wise Learning Rate Decay (LLRD)
   - Mixed precision training
   - Gradient accumulation

2. **Extract features** from fine-tuned BERT (384-dim hidden layer)

3. **Train SVC** on extracted features

4. **Ensemble** predictions from 5-fold cross-validation using majority voting

## Project Structure

```
sentiment-analysis/
├── src/
│   ├── config.py       # Hyperparameters and settings
│   ├── dataset.py      # PyTorch Dataset class
│   ├── model.py        # BERT + ArcFace model
│   ├── trainer.py      # Training loop with LLRD
│   ├── features.py     # Feature extraction
│   └── inference.py    # Prediction pipeline
├── notebooks/
│   ├── bert_sentiment.ipynb                    # Main experiment
│   └── logistic_regression_catboost_baseline.ipynb  # Baselines
├── train.py            # Training script
├── predict.py          # Inference script
└── pyproject.toml
```

## Installation

```bash
# With uv
uv sync

# With pip
pip install -e .

# For GPU-accelerated SVC (optional, requires RAPIDS)
uv sync --extra gpu
```

## Usage

### Training

```bash
# With default ruBERT model
python train.py --data-path data/train.csv --epochs 3 --folds 5

# With another HuggingFace model
python train.py --data-path data/train.csv --model-name bert-base-multilingual-cased
```

### Inference

```bash
# Single text
python predict.py --text "Отличный магазин, всем рекомендую!"

# Batch prediction
python predict.py --input-file data/test.csv --output-file predictions.csv

# With evaluation
python predict.py --input-file data/test.csv --label-column rate

# With another model (must match the model used during training)
python predict.py --text "Great product!" --model-name bert-base-multilingual-cased
```

### Interactive mode

```bash
python predict.py
> Очень плохое обслуживание
Predicted Rating: 1/5
```

## Key Techniques

- **ArcFace Loss**: Adds angular margin to embeddings for better class separation
- **LLRD**: Lower learning rates for lower BERT layers to preserve pretrained knowledge
- **5-Fold CV Ensemble**: Majority voting across folds reduces variance

## Dataset

- ~47K Russian reviews with ratings 1-5
- Imbalanced: 52.6% are 5-star ratings
- Train/Test split: 90/10

To use your own data, provide a CSV with `text` and `rate` columns.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.35+
- CUDA GPU