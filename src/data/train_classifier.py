
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processed" / "alphabet_features.csv"
MODEL_DIR = ROOT / "models"

# Separate model: does not overwrite the existing combined model.
MODEL_PATH = MODEL_DIR / "alphabet_only_classifier.joblib"


def load_alphabet_data():
    features = []
    labels = []

    with DATA_FILE.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            label = row["label"].strip().upper()

            # Keep English alphabet letters only.
            if len(label) != 1 or label not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                continue

            values = [float(value) for value in row["features"].split()]

            if len(values) != 126:
                print(f"Skipping invalid feature row: {row.get('source', '')}")
                continue

            features.append(values)
            labels.append(label)

    return np.asarray(features, dtype=np.float32), np.asarray(labels)


def main():
    if not DATA_FILE.exists():
        print(f"Feature file not found: {DATA_FILE}")
        print("Run src\\data\\extract_features.py first.")
        return

    X, y = load_alphabet_data()

    if len(X) == 0:
        print("No alphabet features found.")
        return

    classes, counts = np.unique(y, return_counts=True)

    print(f"Alphabet samples: {len(X)}")
    print(f"Alphabet classes: {len(classes)}")

    missing = sorted(set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") - set(classes))
    if missing:
        print("Missing classes:", ", ".join(missing))
        print("Training stopped to avoid an incomplete alphabet model.")
        return

    if counts.min() < 2:
        print("Not enough samples in one or more classes.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=42,
        ),
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(f"\nTest accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            zero_division=0,
        )
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nAlphabet-only model saved: {MODEL_PATH}")
    print("Existing combined model has not been changed.")


if __name__ == "__main__":
    main()
