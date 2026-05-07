"""Pipelines listos: preprocesamiento + vectorizacion + modelo."""
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessor import TextPreprocessor
from src.vectorizer import build_vectorizer


def build_pipeline(algorithm: str) -> Pipeline:
    """Construye el pipeline completo segun algoritmo."""
    if algorithm == "naive_bayes":
        return Pipeline([
            ("clean", TextPreprocessor(remove_stopwords=False)),
            ("vec", build_vectorizer("bow")),
            ("clf", MultinomialNB(alpha=1.0)),
        ])
    if algorithm == "logistic":
        return Pipeline([
            ("clean", TextPreprocessor(remove_stopwords=False)),
            ("vec", build_vectorizer("tfidf")),
            ("clf", LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="liblinear",
                class_weight="balanced",
            )),
        ])
    if algorithm == "logistic_char":
        # Char n-grams: el mejor cross-lingual en general.
        return Pipeline([
            ("clean", TextPreprocessor(remove_stopwords=False)),
            ("vec", build_vectorizer("tfidf_char")),
            ("clf", LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="liblinear",
                class_weight="balanced",
            )),
        ])
    raise ValueError(f"Algoritmo desconocido: {algorithm}")


ALGORITHMS = ("naive_bayes", "logistic", "logistic_char")
