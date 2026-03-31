# Russian Sentiment Analysis

Fine-tuned BERT with ArcFace loss for multi-class sentiment classification on text reviews.

## Notebooks

| Notebook                                                   | Description                                     | Link                                                                                                                                                                                                                 |
|------------------------------------------------------------|-------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Fine-tuned BERT ai-forever/ruRoBERTa-large                 | Main solution with ArcFace loss and ensemble    | Current source code                                                                                                                                                                                                  |
| Fine-tuned BERT DeepPavlov/rubert-base-cased-conversational + SVC | Solution with ArcFace loss and ensemble         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/darimorgan/sentiment-analysis/blob/main/notebooks/bert_sentiment.ipynb)                        |
| Baselines                                                  | Logistic Regression & CatBoost with frozen BERT DeepPavlov/rubert-base-cased-conversational| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/darimorgan/sentiment-analysis/blob/main/notebooks/logistic_regression_catboost_baseline.ipynb) |

## Results

| Model                                          | Test F1      |
|------------------------------------------------|--------------|
| Frozen ruBERT + Logistic Regression            | 0.6200       |
| Frozen ruBERT + CatBoost                       | ~0.5972      |
| Fine-tuned ruBERT + SVC Ensemble               | 0.6599       |
| Fine-tuned ruBERT + LogReg Ensemble            | 0.6600       |
| **Fine-tuned ruRoBERTa-large + Direct (2 folds, 3 epochs)** | **0.6840** |

## Approach

1. **Fine-tune BERT** (default: `DeepPavlov/rubert-base-cased-conversational`, configurable via `--model-name`) with:
   - ArcFace loss for better class separation
   - Layer-wise Learning Rate Decay (LLRD)
   - Mixed precision training (AMP)
   - Gradient accumulation
   - Mean pooling + hidden layer (384-dim)

2. **Direct classification** or **extract features** from fine-tuned BERT for SVC/LogReg

3. **Ensemble** predictions from cross-validation folds using majority voting

## Project Structure

```
sentiment-analysis/
├── src/
│   ├── config.py       # Hyperparameters and settings
│   ├── dataset.py      # PyTorch Dataset class
│   ├── model.py        # BERT + ArcFace model
│   ├── trainer.py      # Training loop with LLRD
│   ├── features.py     # Feature extraction
│   ├── mlm.py          # MLM domain pretraining
│   └── inference.py    # Prediction pipeline
├── notebooks/
│   ├── bert_sentiment.ipynb                    # Fine-tuned ruBERT + SVC experiment
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

# For GPU-accelerated SVC/LogReg (optional, requires RAPIDS)
uv sync --extra gpu
```

## Usage

### Training

```bash
# Direct classification with ruRoBERTa-large (best result)
python train.py --data-path data/train.csv --epochs 3 --folds 3 --classifier direct --model-name ai-forever/ruRoBERTa-large --split no --mlm-epochs 0

# With SVC on top of BERT features
python train.py --data-path data/train.csv --classifier svc

# With LogReg instead of SVC
python train.py --data-path data/train.csv --classifier logreg

# With default ruBERT model
python train.py --data-path data/train.csv --model-name DeepPavlov/rubert-base-cased-conversational

# Force specific device
python train.py --data-path data/train.csv --device cuda
python train.py --data-path data/train.csv --device mps
python train.py --data-path data/train.csv --device cpu
```

#### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-path` | required | Path to CSV data |
| `--classifier` | `direct` | Classifier type: `svc`, `logreg`, or `direct` |
| `--num-classes` | `5` | Number of classes |
| `--epochs` | `5` | Number of training epochs |
| `--folds` | `5` | Number of CV folds |
| `--batch-size` | `8` | Batch size |
| `--mlm-epochs` | `2` | MLM pretraining epochs (0 to skip) |
| `--model-name` | `DeepPavlov/rubert-base-cased-conversational` | HuggingFace model |
| `--split` | `yes` | `yes` to split train/test, `no` to use all data |
| `--device` | `auto` | Device: `cuda`, `cpu`, `mps`, or `auto` |

### Inference

```bash
# Single text (direct mode, default)
python predict.py --text "Отличный магазин, всем рекомендую!"

# Batch prediction
python predict.py --input-file data/test.csv --output-file predictions.csv

# With evaluation
python predict.py --input-file data/test.csv --label-column rate

# With another model (must match the model used during training)
python predict.py --text "Great product!" --model-name bert-base-multilingual-cased

# With SVC classifier (must match training)
python predict.py --input-file data/test.csv --classifier svc 

# With different number of classes (must match training)
python predict.py --input-file data/test.csv --num-classes 3

# Force specific device
python predict.py --input-file data/test.csv --device cuda
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
- **CV Ensemble**: Majority voting across folds reduces variance
- **MLM Pretraining**: Optional domain-adaptive masked language model pretraining on task data
- **Direct mode**: Use BERT's own classification head without SVC/LogReg

## Dataset

- ~47K Russian reviews with ratings 1-5
- Imbalanced: 52.6% are 5-star ratings
- Train/Test split: 90/10

To use your own data, provide a CSV with `text` and `rate` columns.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.35+
- CUDA GPU, Apple Silicon (MPS), or CPU