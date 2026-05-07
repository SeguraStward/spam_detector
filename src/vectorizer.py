"""Factoria de vectorizadores. char_wb n-gramas son robustos cross-lingual."""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.preprocessor import STOPWORDS


def build_vectorizer(kind: str = "tfidf"):
    """Devuelve un vectorizador configurado para texto bilingue."""
    if kind == "bow":
        return CountVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words=list(STOPWORDS),
        )
    if kind == "tfidf":
        return TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            stop_words=list(STOPWORDS),
        )
    if kind == "tfidf_char":
        # char_wb: robusto a errores ortograficos y cross-lingual.
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )
    raise ValueError(f"Vectorizador desconocido: {kind}")
