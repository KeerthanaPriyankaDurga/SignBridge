"""
Train an LSTM classifier for ISL word signs.

Reads the .npy files produced by extract_word_features.py and trains
a simple LSTM model using TensorFlow/Keras:

    Input  →  (SEQ_LEN, 126)  per sample
    LSTM   →  64 units
    Dense  →  num_classes (softmax)

Saves the trained model as  models/word_lstm.keras
and a label map as          models/word_labels.npy

Usage:
    python src/data/train_word_lstm.py
"""

from pathlib import Path

import numpy as np

# ── paths ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "word_sequences"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "word_lstm.keras"
LABELS_PATH = MODEL_DIR / "word_labels.npy"


def main():
    # ── load data ────────────────────────────────────────────────
    X_path = DATA_DIR / "word_X.npy"
    y_path = DATA_DIR / "word_y.npy"

    if not X_path.exists() or not y_path.exists():
        print(f"Feature files not found in {DATA_DIR}")
        print("Run extract_word_features.py first.")
        return

    X = np.load(X_path)
    y_raw = np.load(y_path)

    print(f"Loaded: {X.shape[0]} samples, "
          f"{X.shape[1]} frames × {X.shape[2]} features")

    # ── encode labels ────────────────────────────────────────────
    labels = sorted(set(y_raw))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    y = np.array([label_to_idx[label] for label in y_raw])
    num_classes = len(labels)

    print(f"Classes ({num_classes}): {labels}")

    # ── import tensorflow here so it doesn't slow down --help ───
    import tensorflow as tf
    from tensorflow.keras import layers, models

    # ── train/test split ─────────────────────────────────────────
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # ── build model ──────────────────────────────────────────────
    model = models.Sequential([
        layers.Input(shape=(X.shape[1], X.shape[2])),

        # Masking so padded zero-frames are ignored
        layers.Masking(mask_value=0.0),

        # LSTM layer — 64 units is enough for 10 classes with ~200
        # samples.  Keeps it light so training is fast.
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.3),

        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # ── train ────────────────────────────────────────────────────
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=8,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=8,
                restore_best_weights=True,
            ),
        ],
    )

    # ── evaluate ─────────────────────────────────────────────────
    loss, acc = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {acc:.4f}")

    # Per-class accuracy
    preds = model.predict(X_test).argmax(axis=1)
    for i, label in enumerate(labels):
        mask = y_test == i
        if mask.sum() == 0:
            continue
        class_acc = (preds[mask] == i).mean()
        print(f"  {label}: {class_acc:.2%}  ({mask.sum()} samples)")

    # ── save ─────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    np.save(LABELS_PATH, np.array(labels))

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Labels saved: {LABELS_PATH}")


if __name__ == "__main__":
    main()