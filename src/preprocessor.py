"""Limpieza y tokenizacion bilingue (ingles + espanol)."""
import re
import html
from nltk.corpus import stopwords


URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+")
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
HTML_TAG_RE = re.compile(r"<[^>]+>")
NON_WORD_RE = re.compile(r"[^\wáéíóúñü\s-]", flags=re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")

STOPWORDS = set(stopwords.words("english")) | set(stopwords.words("spanish"))


def clean_text(text: str) -> str:
    """Normaliza un texto: HTML, URLs, emails, numeros, puntuacion, mayusculas."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" __url__ ", text)
    text = EMAIL_RE.sub(" __email__ ", text)
    text = NUMBER_RE.sub(" __num__ ", text)
    text = text.lower()
    text = NON_WORD_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    tokens = clean_text(text).split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


class TextPreprocessor:
    """Transformer compatible con sklearn Pipeline."""

    def __init__(self, remove_stopwords: bool = False):
        self.remove_stopwords = remove_stopwords

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.remove_stopwords:
            return [" ".join(tokenize(t, True)) for t in X]
        return [clean_text(t) for t in X]

    def fit_transform(self, X, y=None):
        return self.transform(X)
