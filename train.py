"""Main training script for Russian sentiment analysis."""

import argparse
import gc
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from src.config import Config
from src.dataset import RatingDataset
from src.features import FeatureExtractor
from src.model import StableBertClassifier
from src.trainer import BertTrainer

# Optional: cuML for GPU-accelerated classifiers
try:
    from cuml.svm import SVC
    from cuml.linear_model import LogisticRegression

    USE_CUML = True
except ImportError:
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression

    USE_CUML = False


def train_classifier(features: np.ndarray, targets: np.ndarray, config: Config, classifier_type: str):
    """Train classifier (SVC or LogReg) on extracted BERT features."""
    print(f"Training {classifier_type.upper()}...")

    if classifier_type == "logreg":
        if USE_CUML:
            clf = LogisticRegression(
                C=config.logreg_c,
                penalty=config.logreg_penalty,
                max_iter=config.logreg_max_iter,
            )
        else:
            clf = LogisticRegression(
                C=config.logreg_c,
                penalty=config.logreg_penalty,
                max_iter=config.logreg_max_iter,
            )
    else:  # svc
        if USE_CUML:
            clf = SVC(
                C=config.svc_c,
                kernel=config.svc_kernel,
                gamma="auto",
                probability=False,
                cache_size=config.svc_cache_size,
            )
        else:
            clf = SVC(
                C=config.svc_c,
                kernel=config.svc_kernel,
                gamma="auto",
            )

    clf.fit(features, targets)
    return clf


def main(args: argparse.Namespace) -> None:
    config = Config(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        model_dir=Path(args.model_dir),
        model_name=args.model_name,
        num_classes=args.num_classes,
        num_epochs=args.epochs,
        num_folds=args.folds,
        batch_size=args.batch_size,
        device=args.device,
    )

    print(f"Using device: {config.device}")
    print(f"cuML available: {USE_CUML}")
    print(f"Classifier: {args.classifier}")

    # Load and preprocess data
    print(f"\nLoading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    df = df[[config.text_column, config.label_column]].dropna()
    df = df.drop_duplicates(subset=[config.text_column, config.label_column])

    texts = df[config.text_column].tolist()
    targets = df[config.label_column].values - 1  # Convert to 0-indexed

    print(f"Total samples: {len(texts)}")
    print(f"Class distribution: {np.bincount(targets)}")

    if args.split == "yes":
        # Train/test split
        train_texts, test_texts, train_targets, test_targets = train_test_split(
            texts, targets, test_size=0.1, random_state=42, stratify=targets
        )

        print(f"Train: {len(train_texts)}, Test: {len(test_texts)}")

        train_df = pd.DataFrame(
            {
                config.text_column: train_texts,
                config.label_column: np.array(train_targets) + 1,
            }
        )
        test_df = pd.DataFrame(
            {
                config.text_column: test_texts,
                config.label_column: np.array(test_targets) + 1,
            }
        )
        train_df.to_csv(config.output_dir / "train_split.csv", index=False)
        test_df.to_csv(config.output_dir / "test_split.csv", index=False)
    else:
        train_texts = texts
        train_targets = targets
        print("Using all data for training (no test split).")

    # Initialize
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    trainer = BertTrainer(config)
    feature_extractor = FeatureExtractor(config)

    # Create full dataset
    full_dataset = RatingDataset(
        train_texts, train_targets, tokenizer, config.max_length
    )

    # Cross-validation
    skf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=42)
    folds = list(skf.split(train_texts, train_targets))

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{config.num_folds}")
        print(f"{'='*60}")

        # Data loaders
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        train_loader = DataLoader(
            train_subset, batch_size=config.batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_subset, batch_size=config.batch_size, shuffle=False, num_workers=0
        )

        # Train BERT
        model = StableBertClassifier(
            model_name=config.model_name,
            num_classes=config.num_classes,
            dropout=config.dropout_rate,
            hidden_dim=config.hidden_dim,
        )

        trained_bert = trainer.train_fold(train_loader, val_loader, model, fold_idx)

        # Extract features
        train_features, train_svm_targets = feature_extractor.extract(
            trained_bert, train_loader
        )
        val_features, val_targets_cv = feature_extractor.extract(
            trained_bert, val_loader
        )

        # Train classifier
        clf_model = train_classifier(train_features, train_svm_targets, config, args.classifier)
        joblib.dump(clf_model, config.model_dir / f"fold_{fold_idx}_clf.joblib")

        # Evaluate ensemble
        clf_preds = clf_model.predict(val_features)
        ensemble_f1 = f1_score(val_targets_cv, clf_preds, average="weighted")
        fold_results.append(ensemble_f1)

        print(f"\nFold {fold_idx + 1} Ensemble F1: {ensemble_f1:.4f}")

        # Cleanup
        del trained_bert, clf_model, train_features, val_features
        torch.cuda.empty_cache()
        gc.collect()

    # Final results
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Individual F1 scores: {np.round(fold_results, 4)}")
    print(f"Mean F1: {np.mean(fold_results):.4f} +/- {np.std(fold_results):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sentiment analysis model")
    parser.add_argument("--data-path", type=str, required=True, help="Path to CSV data")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument(
        "--output-dir", type=str, default="outputs", help="Output directory"
    )
    parser.add_argument(
        "--model-dir", type=str, default="models", help="Model directory"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="DeepPavlov/rubert-base-cased-conversational",
        help="Pretrained model name or path",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")

    parser.add_argument(
        "--split",
        type=str,
        choices=["yes", "no"],
        default="yes",
        help="'yes' to split data into train/test (default), 'no' to use all data for training",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=5,
        help="Number of classes (default: 5)",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        choices=["svc", "logreg"],
        default="svc",
        help="Classifier type: 'svc' or 'logreg' (default: svc)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "mps", "auto"],
        default="auto",
        help="Device: 'cuda', 'cpu', 'mps', or 'auto' (default: auto)",
    )

    args = parser.parse_args()
    main(args)
