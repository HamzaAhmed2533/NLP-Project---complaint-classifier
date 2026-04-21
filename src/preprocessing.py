"""
preprocessing.py
----------------
Implements a reusable and modular text preprocessing pipeline for
customer complaint classification.

Steps (all configurable via PreprocessingConfig):
    1. Lowercase
    2. Remove URLs, emails, numbers, punctuation, extra whitespace
    3. Tokenisation
    4. Stopword removal
    5. Stemming  (optional)
    6. Lemmatisation (optional — cannot be combined with stemming)

Usage
-----
    from src.preprocessing import PreprocessingConfig, TextPreprocessor

    config = PreprocessingConfig(use_lemmatization=True)
    preprocessor = TextPreprocessor(config)
    clean_texts = preprocessor.transform(df["text"])
"""

import re
import string
from dataclasses import dataclass, field
from typing import List, Optional

import nltk
import pandas as pd


# ---------------------------------------------------------------------------
# NLTK resource bootstrap
# ---------------------------------------------------------------------------

def _ensure_nltk_resources() -> None:
    """Download required NLTK data packages if not already present."""
    resources = [
        ("tokenizers/punkt",          "punkt"),
        ("tokenizers/punkt_tab",      "punkt_tab"),
        ("corpora/stopwords",         "stopwords"),
        ("corpora/wordnet",           "wordnet"),
        ("corpora/omw-1.4",           "omw-1.4"),
    ]
    for resource_path, resource_id in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"[preprocessing] Downloading NLTK resource: {resource_id}")
            nltk.download(resource_id, quiet=True)


_ensure_nltk_resources()

from nltk.corpus import stopwords                    # noqa: E402
from nltk.stem import PorterStemmer, WordNetLemmatizer  # noqa: E402
from nltk.tokenize import word_tokenize              # noqa: E402


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PreprocessingConfig:
    """
    Configuration object that controls which preprocessing steps are applied.

    Attributes
    ----------
    lowercase : bool
        Convert all text to lowercase.
    remove_urls : bool
        Strip HTTP/HTTPS URLs from text.
    remove_emails : bool
        Strip email addresses from text.
    remove_numbers : bool
        Remove standalone numeric tokens.
    remove_punctuation : bool
        Remove punctuation characters.
    remove_stopwords : bool
        Remove common English stopwords.
    use_stemming : bool
        Apply Porter stemmer to tokens.
    use_lemmatization : bool
        Apply WordNet lemmatizer to tokens. Cannot be combined with stemming.
    extra_stopwords : List[str]
        Additional domain-specific words to treat as stopwords.
    """
    lowercase: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_numbers: bool = True
    remove_punctuation: bool = True
    remove_stopwords: bool = True
    use_stemming: bool = False
    use_lemmatization: bool = True
    extra_stopwords: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.use_stemming and self.use_lemmatization:
            raise ValueError(
                "use_stemming and use_lemmatization cannot both be True. "
                "Choose one."
            )


# ---------------------------------------------------------------------------
# Preprocessor class
# ---------------------------------------------------------------------------

class TextPreprocessor:
    """
    Applies the full text preprocessing pipeline to a list of strings
    or a pandas Series.

    Parameters
    ----------
    config : PreprocessingConfig
        Controls which preprocessing steps are active.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        self.config = config or PreprocessingConfig()
        self._stop_words = self._build_stopword_set()
        self._stemmer = PorterStemmer() if self.config.use_stemming else None
        self._lemmatizer = (
            WordNetLemmatizer() if self.config.use_lemmatization else None
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess a single raw text string and return a cleaned string.

        Parameters
        ----------
        text : str
            Raw customer complaint string.

        Returns
        -------
        str
            Cleaned, preprocessed string (tokens joined by spaces).
        """
        if not isinstance(text, str):
            text = str(text)

        if self.config.lowercase:
            text = text.lower()

        if self.config.remove_urls:
            text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        if self.config.remove_emails:
            text = re.sub(r"\S+@\S+", " ", text)

        if self.config.remove_numbers:
            text = re.sub(r"\b\d+\b", " ", text)

        if self.config.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        # Tokenise
        tokens: List[str] = word_tokenize(text)

        # Stopword removal
        if self.config.remove_stopwords:
            tokens = [t for t in tokens if t not in self._stop_words]

        # Remove residual single-character tokens and blank strings
        tokens = [t for t in tokens if len(t) > 1]

        # Stemming / Lemmatization
        if self._stemmer:
            tokens = [self._stemmer.stem(t) for t in tokens]
        elif self._lemmatizer:
            tokens = [self._lemmatizer.lemmatize(t) for t in tokens]

        return " ".join(tokens)

    def transform(self, texts: pd.Series) -> pd.Series:
        """
        Apply preprocessing to every entry in a pandas Series.

        Parameters
        ----------
        texts : pd.Series
            Series of raw complaint strings.

        Returns
        -------
        pd.Series
            Series of preprocessed strings with the same index.
        """
        print(f"[preprocessing] Processing {len(texts)} texts ...")
        cleaned = texts.apply(self.preprocess_text)
        print("[preprocessing] Done.")
        return cleaned

    def describe(self) -> None:
        """Print the active preprocessing configuration."""
        print("\n[preprocessing] Active configuration:")
        for key, val in vars(self.config).items():
            print(f"  {key:<25} = {val}")
        print()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_stopword_set(self) -> set:
        """Build combined stopword set from NLTK + user-supplied extras."""
        stop = set(stopwords.words("english"))
        stop.update(self.config.extra_stopwords)
        return stop
