"""Entrena, evalua y persiste los modelos de deteccion de spam bilingue."""
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from src.evaluator import evaluate, print_report
from src.load_dataset import load_all
from src.models import ALGORITHMS, build_pipeline

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def model_path(algorithm: str) -> Path:
    return MODELS_DIR / f"{algorithm}.joblib"


def train_and_save(algorithm: str, X_train, y_train) -> object:
    pipeline = build_pipeline(algorithm)
    pipeline.fit(X_train, y_train)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = model_path(algorithm)
    joblib.dump(pipeline, path)
    print(f"[ok] Modelo guardado en {path}")
    return pipeline


def load_model(algorithm: str):
    path = model_path(algorithm)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Entrena primero: python -m src.train"
        )
    return joblib.load(path)


def run() -> dict[str, dict]:
    df = load_all()
    print(f"Dataset combinado: {len(df)} mensajes")
    print(df.groupby(["lang", "label"]).size().unstack(fill_value=0))

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )
    print(f"\nTrain: {len(X_train)}  Test: {len(X_test)}")

    results = {}
    for algorithm in ALGORITHMS:
        print(f"\n>>> Entrenando {algorithm} ...")
        pipeline = train_and_save(algorithm, X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = evaluate(y_test, y_pred)
        print_report(algorithm, metrics)
        results[algorithm] = metrics
    return results


if __name__ == "__main__":
    run()
