"""Metricas y reporte de evaluacion."""
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate(y_true, y_pred, positive_label: str = "spam") -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=["ham", "spam"]).tolist(),
        "report": classification_report(y_true, y_pred, zero_division=0),
    }


def print_report(name: str, metrics: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}  (sobre clase 'spam')")
    print(f"Recall:    {metrics['recall']:.4f}  (sobre clase 'spam')")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print("\nMatriz de confusion (filas=real, cols=pred) [ham, spam]:")
    cm = metrics["confusion_matrix"]
    print(f"          pred_ham  pred_spam")
    print(f"real_ham  {cm[0][0]:8d}  {cm[0][1]:9d}")
    print(f"real_spam {cm[1][0]:8d}  {cm[1][1]:9d}")
    print("\nReporte detallado:")
    print(metrics["report"])
