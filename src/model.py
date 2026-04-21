"""
model.py
--------
Defines and trains machine learning classifiers for customer complaint
classification.

Supported models:
    - Logistic Regression  (primary)
    - Naive Bayes          (baseline — MultinomialNB)
    - Support Vector Machine (SVM — LinearSVC, optional bonus)

Each model is wrapped in a ModelTrainer that stores the fitted estimator
and provides a consistent train / predict API.

Usage
-----
    from src.model import ModelTrainer, ModelConfig

    config = ModelConfig(model_type="logistic_regression", random_state=42)
    trainer = ModelTrainer(config)
    trainer.train(X_train, y_train)
    predictions = trainer.predict(X_test)
"""

from dataclasses import dataclass, field
from typing import Any, Dict

import joblib
from scipy.sparse import spmatrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


# ---------------------------------------------------------------------------
# Supported model registry
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: Dict[str, Any] = {
    "logistic_regression": LogisticRegression,
    "naive_bayes": MultinomialNB,
    "svm": LinearSVC,
}

SUPPORTED_MODELS = list(_MODEL_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    Configuration for a classifier.

    Attributes
    ----------
    model_type : str
        One of 'logistic_regression', 'naive_bayes', or 'svm'.
    random_state : int
        Random seed for reproducibility (used by models that support it).
    max_iter : int
        Maximum number of iterations for solvers that accept it (LR, SVM).
    model_kwargs : dict
        Any additional keyword arguments forwarded directly to the sklearn
        estimator constructor.
    """
    model_type: str = "logistic_regression"
    random_state: int = 42
    max_iter: int = 1000
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model_type not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unsupported model_type '{self.model_type}'. "
                f"Choose from: {SUPPORTED_MODELS}"
            )


# ---------------------------------------------------------------------------
# ModelTrainer class
# ---------------------------------------------------------------------------

class ModelTrainer:
    """
    Wraps an sklearn classifier with a simple train / predict interface.

    Parameters
    ----------
    config : ModelConfig
        Specifies which model to build and its hyperparameters.
    """

    def __init__(self, config: ModelConfig = None) -> None:
        self.config = config or ModelConfig()
        self.model = self._build_model()
        self._is_trained = False

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def train(self, X_train: spmatrix, y_train) -> None:
        """
        Fit the classifier on the provided feature matrix and labels.

        Parameters
        ----------
        X_train : sparse matrix or array-like
            Feature matrix for training samples.
        y_train : array-like
            Target labels for training samples.
        """
        model_name = self.config.model_type.replace("_", " ").title()
        print(f"[model] Training '{model_name}' ...")
        self.model.fit(X_train, y_train)
        self._is_trained = True
        print(f"[model] Training complete.")

    def predict(self, X):
        """
        Generate class predictions for the given feature matrix.

        Parameters
        ----------
        X : sparse matrix or array-like
            Feature matrix for samples to predict.

        Returns
        -------
        np.ndarray
            Predicted class labels.

        Raises
        ------
        RuntimeError
            If called before train().
        """
        self._check_trained()
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Generate class probability estimates (only for models that support it).

        Parameters
        ----------
        X : sparse matrix or array-like
            Feature matrix for samples.

        Returns
        -------
        np.ndarray
            Probability matrix of shape (n_samples, n_classes), or None if
            the model does not support probability estimates.
        """
        self._check_trained()
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def save(self, filepath: str) -> None:
        """
        Persist the trained model to disk using joblib.

        Parameters
        ----------
        filepath : str
            Destination path (e.g. 'models/lr_model.joblib').
        """
        self._check_trained()
        joblib.dump(self.model, filepath)
        print(f"[model] Model saved to '{filepath}'.")

    @staticmethod
    def load(filepath: str) -> "ModelTrainer":
        """
        Load a previously saved model from disk.

        Parameters
        ----------
        filepath : str
            Path to a joblib-serialised sklearn estimator.

        Returns
        -------
        ModelTrainer
            A ModelTrainer instance wrapping the loaded estimator.
        """
        trainer = ModelTrainer.__new__(ModelTrainer)
        trainer.model = joblib.load(filepath)
        trainer._is_trained = True
        trainer.config = ModelConfig(model_type="logistic_regression")  # placeholder
        print(f"[model] Model loaded from '{filepath}'.")
        return trainer

    def describe(self) -> None:
        """Print the active model configuration."""
        print("\n[model] Active configuration:")
        for key, val in vars(self.config).items():
            print(f"  {key:<20} = {val}")
        print(f"  {'estimator':<20} = {self.model.__class__.__name__}")
        print()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_model(self):
        """Instantiate the sklearn estimator based on config."""
        cls = _MODEL_REGISTRY[self.config.model_type]
        kwargs = dict(self.config.model_kwargs)

        # Inject seed / iteration params only for models that accept them
        if self.config.model_type == "logistic_regression":
            kwargs.setdefault("random_state", self.config.random_state)
            kwargs.setdefault("max_iter", self.config.max_iter)

        elif self.config.model_type == "svm":
            kwargs.setdefault("random_state", self.config.random_state)
            kwargs.setdefault("max_iter", self.config.max_iter)

        # MultinomialNB has no random_state / max_iter parameter
        return cls(**kwargs)

    def _check_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError(
                "Model has not been trained yet. Call train() first."
            )
