from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def evaluate_predictions(y_true, y_pred, labels=None):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }


if __name__ == "__main__":
    print("Evaluation module ready.")
    print("Metrics will be calculated from real test predictions.")
