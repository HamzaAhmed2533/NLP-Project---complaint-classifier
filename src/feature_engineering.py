"""
feature_engineering.py
-----------------------
Converts preprocessed text into numerical feature representations using:
    1. Bag of Words  (CountVectorizer)
    2. TF-IDF        (TfidfVectorizer)

Both vectorisers are wrapped in a FeatureExtractor class that provides a
consistent fit / transform / fit_transform API and easy configuration.

Usage
-----
    from src.feature_engineering import FeatureExtractor, FeatureConfig

    config = FeatureConfig(method="tfidf", max_features=5000, ngram_range=(1, 2))
    extractor = FeatureExtractor(config)
    X_train = extractor.fit_transform(train_texts)
    X_test  = extractor.transform(test_texts)
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """
    Configuration for feature extraction.

    Attributes
    ----------
    method : str
        Either 'bow' (Bag of Words) or 'tfidf'.
    max_features : int or None
        Maximum number of vocabulary tokens to keep (most frequent).
        None means unlimited.
    ngram_range : tuple
        Lower and upper boundary of the n-gram size, e.g. (1, 1) for
        unigrams only, (1, 2) for unigrams and bigrams.
    min_df : int or float
        Minimum document frequency for a token to be included.
    max_df : float
        Maximum document frequency ratio for a token to be included.
        Useful for filtering corpus-wide common words.
    """
    method: str = "tfidf"          # 'bow' | 'tfidf'
    max_features: int = 10_000
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 1
    max_df: float = 0.95

    def __post_init__(self) -> None:
        allowed = {"bow", "tfidf"}
        if self.method not in allowed:
            raise ValueError(
                f"method must be one of {allowed}, got '{self.method}'."
            )


# ---------------------------------------------------------------------------
# FeatureExtractor class
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Wraps sklearn vectorisers to provide a unified API for feature extraction.

    Parameters
    ----------
    config : FeatureConfig
        Controls which vectoriser is used and its hyperparameters.
    """

    def __init__(self, config: FeatureConfig = None) -> None:
        self.config = config or FeatureConfig()
        self._vectorizer = self._build_vectorizer()
        self.fitted = False

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def fit_transform(self, texts: pd.Series) -> spmatrix:
        """
        Fit the vectoriser on training text and return the feature matrix.

        Parameters
        ----------
        texts : pd.Series
            Preprocessed training texts.

        Returns
        -------
        scipy.sparse matrix
            Sparse feature matrix of shape (n_samples, n_features).
        """
        print(
            f"[feature_engineering] Fitting '{self.config.method}' vectoriser "
            f"(max_features={self.config.max_features}, "
            f"ngram_range={self.config.ngram_range}) ..."
        )
        X = self._vectorizer.fit_transform(texts)
        self.fitted = True
        print(
            f"[feature_engineering] Feature matrix shape: {X.shape} "
            f"({X.shape[0]} samples × {X.shape[1]} features)"
        )
        return X

    def transform(self, texts: pd.Series) -> spmatrix:
        """
        Transform text using the already-fitted vectoriser.

        Parameters
        ----------
        texts : pd.Series
            Preprocessed texts (train or test split).

        Returns
        -------
        scipy.sparse matrix
            Sparse feature matrix.

        Raises
        ------
        RuntimeError
            If called before fit_transform().
        """
        if not self.fitted:
            raise RuntimeError(
                "FeatureExtractor has not been fitted yet. "
                "Call fit_transform() on the training data first."
            )
        return self._vectorizer.transform(texts)

    def get_feature_names(self) -> np.ndarray:
        """Return the vocabulary feature names as a numpy array."""
        if not self.fitted:
            raise RuntimeError("FeatureExtractor has not been fitted yet.")
        return np.array(self._vectorizer.get_feature_names_out())

    def describe(self) -> None:
        """Print the active feature extraction configuration."""
        print("\n[feature_engineering] Active configuration:")
        for key, val in vars(self.config).items():
            print(f"  {key:<20} = {val}")
        print()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_vectorizer(self):
        """Instantiate the correct sklearn vectoriser based on config."""
        common_kwargs = dict(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
        )
        if self.config.method == "tfidf":
            return TfidfVectorizer(**common_kwargs)
        else:  # 'bow'
            return CountVectorizer(**common_kwargs)
