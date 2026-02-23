"""Inference script for sentiment prediction."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

from src.config import Config
from src.inference import SentimentPredictor


def main(args: argparse.Namespace) -> None:
    config = Config(
        model_dir=Path(args.model_dir),
        model_name=args.model_name,
    )

    print(f"Using device: {config.device}")
    print(f"Model: {config.model_name}")

    # Initialize predictor
    predictor = SentimentPredictor(config, model_dir=args.model_dir)
    predictor.load_models(num_folds=args.num_folds)

    if args.input_file:
        # Batch prediction from CSV
        print(f"\nLoading data from {args.input_file}...")
        df = pd.read_csv(args.input_file)

        texts = df[args.text_column].tolist()
        predictions = predictor.predict(texts, batch_size=args.batch_size)

        # Save predictions
        df["predicted_rating"] = predictions
        output_path = (
            Path(args.output_file) if args.output_file else Path("predictions.csv")
        )
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

        # If labels exist, compute metrics
        if args.label_column and args.label_column in df.columns:
            true_labels = df[args.label_column].values
            f1 = f1_score(true_labels, predictions, average="weighted")
            print(f"\nWeighted F1 Score: {f1:.4f}")

    elif args.text:
        # Single text prediction
        rating = predictor.predict_single(args.text)
        print(f"\nText: {args.text}")
        print(f"Predicted Rating: {rating}/5")

    else:
        # Interactive mode
        print("\nInteractive mode. Enter text to predict (Ctrl+C to exit):")
        while True:
            try:
                text = input("\n> ")
                if text.strip():
                    rating = predictor.predict_single(text)
                    print(f"Predicted Rating: {rating}/5")
            except KeyboardInterrupt:
                print("\nExiting...")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict sentiment ratings")
    parser.add_argument(
        "--model-dir", type=str, default="models", help="Directory with trained models"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="DeepPavlov/rubert-base-cased-conversational",
        help="Pretrained model name or path",
    )
    parser.add_argument(
        "--num-folds", type=int, default=5, help="Number of ensemble folds"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for inference"
    )

    # Input options (mutually exclusive in practice)
    parser.add_argument("--input-file", type=str, help="CSV file with texts to predict")
    parser.add_argument("--text", type=str, help="Single text to predict")
    parser.add_argument(
        "--text-column", type=str, default="text", help="Column name for text"
    )
    parser.add_argument(
        "--label-column", type=str, help="Column name for labels (for evaluation)"
    )
    parser.add_argument(
        "--output-file", type=str, help="Output CSV path for predictions"
    )

    args = parser.parse_args()
    main(args)
