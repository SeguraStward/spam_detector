"""CLI del detector de spam bilingue.

Uso:
    python -m src.main --train
    python -m src.main --algorithm logistic --message "WIN a free iPhone now!!!"
    python -m src.main --algorithm naive_bayes --file correo.txt
    python -m src.main --algorithm logistic_char --evaluate
"""
import argparse
from pathlib import Path

from src.evaluator import evaluate, print_report
from src.load_dataset import load_all
from src.models import ALGORITHMS
from src.train import load_model, run as train_all
from sklearn.model_selection import train_test_split


def predict_message(algorithm: str, text: str) -> dict:
    model = load_model(algorithm)
    pred = model.predict([text])[0]
    proba = None
    if hasattr(model, "predict_proba"):
        classes = model.classes_
        probs = model.predict_proba([text])[0]
        proba = {cls: float(p) for cls, p in zip(classes, probs)}
        confidence = float(max(probs))
    else:
        confidence = 1.0
    return {"prediction": pred, "confidence": confidence, "probabilities": proba}


def cmd_predict(algorithm: str, text: str) -> None:
    result = predict_message(algorithm, text)
    print(f"\nAlgoritmo:   {algorithm}")
    print(f"Prediccion:  {result['prediction'].upper()}")
    print(f"Confianza:   {result['confidence']:.2%}")
    if result["probabilities"]:
        print("Probabilidades:")
        for cls, p in result["probabilities"].items():
            print(f"  P({cls}) = {p:.2%}")


def cmd_evaluate(algorithm: str) -> None:
    df = load_all()
    _, X_test, _, y_test = train_test_split(
        df["text"].tolist(), df["label"].tolist(),
        test_size=0.2, random_state=42, stratify=df["label"],
    )
    model = load_model(algorithm)
    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred)
    print_report(algorithm, metrics)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detector de spam bilingue (ingles + espanol)"
    )
    parser.add_argument(
        "--algorithm", "-a", choices=ALGORITHMS, default="logistic",
        help="Algoritmo a usar (default: logistic)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--message", "-m", help="Mensaje a clasificar")
    group.add_argument("--file", "-f", help="Ruta a archivo .txt con el mensaje")
    group.add_argument("--evaluate", "-e", action="store_true",
                       help="Evaluar modelo sobre el test set")
    group.add_argument("--train", "-t", action="store_true",
                       help="Entrenar y persistir todos los modelos")

    args = parser.parse_args()

    if args.train:
        train_all()
        return
    if args.evaluate:
        cmd_evaluate(args.algorithm)
        return
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8", errors="ignore")
        cmd_predict(args.algorithm, text)
        return
    if args.message:
        cmd_predict(args.algorithm, args.message)
        return
    parser.print_help()


if __name__ == "__main__":
    main()
