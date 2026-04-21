"""
main.py
-------
End-to-end pipeline runner for the Customer Complaint Classification system.

Run this script from the project root:
    python main.py

Optional command-line arguments:
    --dataset    PATH      Path to CSV dataset  (default: data/complaints.csv)
    --feature    METHOD    'bow' or 'tfidf'     (default: tfidf)
    --model      NAME      Model to run:        (default: all)
                           logistic_regression | naive_bayes | svm | all
    --ngrams     N M       n-gram range         (default: 1 2)
    --max-feat   N         Max vocabulary size  (default: 10000)
    --save-models          Save trained models to models/ directory
    --no-plots             Skip confusion matrix plots

Example:
    python main.py --feature tfidf --model logistic_regression --save-models
"""

import argparse
import io
import os
import sys

# Force UTF-8 output on Windows so Unicode characters render correctly
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path when running as a script
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.model_selection import train_test_split

from src.data_loader import load_data, inspect_data
from src.evaluation import Evaluator
from src.feature_engineering import FeatureConfig, FeatureExtractor
from src.model import ModelConfig, ModelTrainer, SUPPORTED_MODELS
from src.preprocessing import PreprocessingConfig, TextPreprocessor


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DATASET = os.path.join(PROJECT_ROOT, "data", "complaints.csv")
TEST_SIZE       = 0.20   # 80 % train – 20 % test split
RANDOM_STATE    = 42


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Customer Complaint Classification — NLP Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Path to the CSV dataset (must have 'text' and 'label' columns).",
    )
    parser.add_argument(
        "--feature",
        choices=["bow", "tfidf"],
        default="tfidf",
        help="Feature extraction method (default: tfidf).",
    )
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS + ["all"],
        default="all",
        help="Model to train. Use 'all' to run every model (default: all).",
    )
    parser.add_argument(
        "--ngrams",
        nargs=2,
        type=int,
        metavar=("N", "M"),
        default=[1, 2],
        help="n-gram range, e.g. --ngrams 1 2  (default: 1 2).",
    )
    parser.add_argument(
        "--max-feat",
        type=int,
        default=10_000,
        dest="max_feat",
        help="Maximum vocabulary size (default: 10000).",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        dest="save_models",
        help="Persist trained models to the models/ directory.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        dest="no_plots",
        help="Skip confusion matrix visualisations.",
    )
    return parser


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage_load(dataset_path: str):
    """Load and inspect the raw dataset."""
    print("\n" + "━" * 60)
    print("  STAGE 1 — DATA LOADING")
    print("━" * 60)
    df = load_data(dataset_path)
    inspect_data(df)
    return df


def stage_preprocess(df):
    """Clean and normalise complaint text."""
    print("\n" + "━" * 60)
    print("  STAGE 2 — TEXT PREPROCESSING")
    print("━" * 60)
    config = PreprocessingConfig(use_lemmatization=True, use_stemming=False)
    preprocessor = TextPreprocessor(config)
    preprocessor.describe()
    df["clean_text"] = preprocessor.transform(df["text"])
    return df


def stage_split(df):
    """Split dataset into training and test sets."""
    print("\n" + "━" * 60)
    print("  STAGE 3 — TRAIN / TEST SPLIT")
    print("━" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    print(f"  Training samples : {len(X_train)}")
    print(f"  Test samples     : {len(X_test)}")
    return X_train, X_test, y_train, y_test


def stage_features(X_train, X_test, feature_method, ngram_range, max_features):
    """Extract numerical features from text."""
    print("\n" + "━" * 60)
    print("  STAGE 4 — FEATURE EXTRACTION")
    print("━" * 60)
    feat_config = FeatureConfig(
        method=feature_method,
        ngram_range=tuple(ngram_range),
        max_features=max_features,
    )
    extractor = FeatureExtractor(feat_config)
    extractor.describe()
    X_train_feat = extractor.fit_transform(X_train)
    X_test_feat  = extractor.transform(X_test)
    return X_train_feat, X_test_feat, extractor


def stage_train_evaluate(
    model_types,
    X_train_feat,
    X_test_feat,
    y_train,
    y_test,
    X_test_raw,
    labels,
    save_models: bool,
    no_plots: bool,
):
    """Train each requested model and evaluate it."""
    print("\n" + "━" * 60)
    print("  STAGE 5 — MODEL TRAINING & EVALUATION")
    print("━" * 60)

    evaluator = Evaluator(labels=labels)
    all_results = {}

    models_dir = os.path.join(PROJECT_ROOT, "models")

    for model_type in model_types:
        model_name = model_type.replace("_", " ").title()

        # --- Train -------------------------------------------------------
        trainer = ModelTrainer(
            ModelConfig(model_type=model_type, random_state=RANDOM_STATE)
        )
        trainer.describe()
        trainer.train(X_train_feat, y_train)

        # --- Predict -----------------------------------------------------
        y_pred = trainer.predict(X_test_feat)

        # --- Evaluate ----------------------------------------------------
        metrics = evaluator.evaluate(y_test, y_pred, model_name=model_name)
        all_results[model_name] = metrics

        # --- Sample predictions ------------------------------------------
        evaluator.show_sample_predictions(
            texts=X_test_raw, y_true=y_test, y_pred=y_pred, n=5
        )

        # --- Confusion matrix --------------------------------------------
        if not no_plots:
            cm_path = os.path.join(
                PROJECT_ROOT,
                "outputs",
                f"confusion_matrix_{model_type}.png",
            )
            evaluator.plot_confusion_matrix(
                y_test, y_pred,
                model_name=model_name,
                save_path=cm_path,
            )

        # --- Save model --------------------------------------------------
        if save_models:
            os.makedirs(models_dir, exist_ok=True)
            save_path = os.path.join(models_dir, f"{model_type}.joblib")
            trainer.save(save_path)

    # --- Comparison table ------------------------------------------------
    if len(all_results) > 1:
        evaluator.compare_results(all_results)

    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_arg_parser()
    args   = parser.parse_args()

    # Determine which models to run
    model_types = (
        SUPPORTED_MODELS if args.model == "all" else [args.model]
    )

    print("\n" + "=" * 60)
    print("  CUSTOMER COMPLAINT CLASSIFICATION — NLP PIPELINE")
    print("=" * 60)
    print(f"  Dataset    : {args.dataset}")
    print(f"  Features   : {args.feature.upper()}")
    print(f"  Models     : {', '.join(m.replace('_', ' ').title() for m in model_types)}")
    print(f"  n-grams    : {tuple(args.ngrams)}")
    print(f"  Max feats  : {args.max_feat}")
    print(f"  Save models: {args.save_models}")

    # Stage 1 — Load
    df = stage_load(args.dataset)

    # Stage 2 — Preprocess
    df = stage_preprocess(df)

    # Stage 3 — Split
    X_train, X_test, y_train, y_test = stage_split(df)

    # Stage 4 — Feature extraction
    X_train_feat, X_test_feat, extractor = stage_features(
        X_train, X_test,
        feature_method=args.feature,
        ngram_range=args.ngrams,
        max_features=args.max_feat,
    )

    # Stage 5 — Train & evaluate
    labels = sorted(df["label"].unique().tolist())
    stage_train_evaluate(
        model_types=model_types,
        X_train_feat=X_train_feat,
        X_test_feat=X_test_feat,
        y_train=y_train,
        y_test=y_test,
        X_test_raw=X_test,
        labels=labels,
        save_models=args.save_models,
        no_plots=args.no_plots,
    )

    print("\n✓ Pipeline completed successfully.\n")


if __name__ == "__main__":
    main()
