"""
evaluation.py
-------------
Provides comprehensive evaluation utilities for the classification models.

Features:
    - Accuracy score
    - Full precision / recall / F1 classification report
    - Per-class breakdown table
    - Confusion matrix (printed as text and saved as a PNG figure)
    - Sample prediction display

Usage
-----
    from src.evaluation import Evaluator

    evaluator = Evaluator(labels=["billing_issue", "delivery_problem", ...])
    evaluator.evaluate(y_true, y_pred, model_name="Logistic Regression")
    evaluator.plot_confusion_matrix(y_true, y_pred, save_path="confusion.png")
    evaluator.show_sample_predictions(texts, y_true, y_pred, n=5)
"""

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Evaluator class
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Computes and displays classification performance metrics.

    Parameters
    ----------
    labels : list of str, optional
        Ordered list of class label names. Improves formatting of reports
        and confusion matrices.
    """

    def __init__(self, labels: Optional[List[str]] = None) -> None:
        self.labels = labels

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        y_true,
        y_pred,
        model_name: str = "Model",
    ) -> dict:
        """
        Compute and print classification metrics.

        Parameters
        ----------
        y_true : array-like
            Ground-truth labels.
        y_pred : array-like
            Predicted labels.
        model_name : str
            Display name used in the printed header.

        Returns
        -------
        dict
            Dictionary containing 'accuracy' and 'report' keys.
        """
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.labels,
            digits=4,
            zero_division=0,
        )

        self._print_header(model_name)
        print(f"  Accuracy : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
        print()
        print("  Classification Report")
        print("  " + "-" * 58)
        for line in report.splitlines():
            print("  " + line)
        print("=" * 60)

        return {"accuracy": accuracy, "report": report}

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------

    def plot_confusion_matrix(
        self,
        y_true,
        y_pred,
        model_name: str = "Model",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Generate and display a confusion matrix heatmap.

        Parameters
        ----------
        y_true : array-like
            Ground-truth labels.
        y_pred : array-like
            Predicted labels.
        model_name : str
            Title shown above the confusion matrix.
        save_path : str, optional
            If provided, saves the figure to this file path.
        """
        cm = confusion_matrix(y_true, y_pred, labels=self.labels)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.labels or "auto",
            yticklabels=self.labels or "auto",
            ax=ax,
            linewidths=0.5,
            linecolor="lightgrey",
        )
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, pad=15)
        plt.xticks(rotation=30, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[evaluation] Confusion matrix saved to '{save_path}'.")

        plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------
    # Sample predictions
    # ------------------------------------------------------------------

    def show_sample_predictions(
        self,
        texts,
        y_true,
        y_pred,
        n: int = 5,
    ) -> None:
        """
        Print n randomly selected predictions alongside the ground truth.

        Parameters
        ----------
        texts : pd.Series or list
            Original (raw or preprocessed) text strings.
        y_true : array-like
            Ground-truth labels.
        y_pred : array-like
            Predicted labels.
        n : int
            Number of samples to display.
        """
        import random
        indices = list(range(len(y_true)))
        random.seed(42)
        sample_idx = random.sample(indices, min(n, len(indices)))

        print("\n" + "=" * 60)
        print(f"SAMPLE PREDICTIONS  (showing {len(sample_idx)} examples)")
        print("=" * 60)

        texts_list = list(texts)
        y_true_list = list(y_true)
        y_pred_list = list(y_pred)

        for rank, idx in enumerate(sample_idx, start=1):
            text_preview = str(texts_list[idx])[:80]
            if len(str(texts_list[idx])) > 80:
                text_preview += "..."
            true_label = y_true_list[idx]
            pred_label = y_pred_list[idx]
            match = "✓" if true_label == pred_label else "✗"

            print(f"\n[{rank}] {match}")
            print(f"  Text    : {text_preview}")
            print(f"  True    : {true_label}")
            print(f"  Predicted: {pred_label}")

        print("=" * 60)

    # ------------------------------------------------------------------
    # Comparison helper
    # ------------------------------------------------------------------

    def compare_results(self, results: dict) -> None:
        """
        Print a side-by-side accuracy comparison for multiple models.

        Parameters
        ----------
        results : dict
            Mapping of model_name -> evaluate() return dict, e.g.:
            {'Logistic Regression': {'accuracy': 0.92, 'report': ...}, ...}
        """
        print("\n" + "=" * 60)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 60)
        print(f"  {'Model':<30} {'Accuracy':>10}")
        print("  " + "-" * 42)
        for model_name, metrics in sorted(
            results.items(), key=lambda x: x[1]["accuracy"], reverse=True
        ):
            acc = metrics["accuracy"]
            print(f"  {model_name:<30} {acc:>10.4f}  ({acc*100:.2f}%)")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _print_header(model_name: str) -> None:
        print("\n" + "=" * 60)
        print(f"EVALUATION RESULTS — {model_name.upper()}")
        print("=" * 60)
