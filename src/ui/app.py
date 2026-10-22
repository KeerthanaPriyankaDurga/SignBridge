"""
SignBridge Flask backend.

Serves the frontend and exposes a /predict endpoint that:
  1. Receives a base64-encoded video frame from the browser
  2. Runs MediaPipe hand detection
  3. Normalises landmarks (wrist-relative + scale) — same pipeline
     that extract_features.py used at training time
  4. Feeds the 126-feature vector into the SVM classifier
  5. Returns { prediction, confidence, mode } as JSON

Also keeps a server-side SentenceBuilder so the browser can
POST /sentence actions (add, space, backspace, clear, get).
"""

import base64
import sys
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, render_template, request

# ── paths ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]          # project root
MODEL_PATH = ROOT / "models" / "alphadigit_classifier.joblib"

# Let Python find the sentence_builder package
sys.path.insert(0, str(ROOT))
from src.sentence_builder.builder import SentenceBuilder

# ── Flask app ──────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "templates"),
)

# ── globals loaded once on startup ─────────────────────────────
model = None
hands = None
builder = SentenceBuilder()


def load_model():
    global model, hands
    if not MODEL_PATH.exists():
        print(f"[SignBridge] Model not found: {MODEL_PATH}")
        print("  Run train_classifier.py first (with digit fix).")
        return False
    model = joblib.load(MODEL_PATH)
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    print(f"[SignBridge] Model loaded — classes: {list(model.classes_)}")
    return True


# ── feature extraction (matches extract_features.py EXACTLY) ──

def normalize_hand(hand):
    """21 landmarks → 63 wrist-relative, scale-normalised floats."""
    wrist = hand.landmark[0]
    points = []
    for pt in hand.landmark:
        points.extend([
            pt.x - wrist.x,
            pt.y - wrist.y,
            pt.z - wrist.z,
        ])
    scale = max(abs(v) for v in points)
    if scale > 0:
        points = [v / scale for v in points]
    return points


def extract_features(hand_landmarks):
    """Up to 2 hands → 1×126 feature array."""
    features = []
    for hand in hand_landmarks[:2]:
        features.extend(normalize_hand(hand))
    features = features[:126]
    features.extend([0.0] * (126 - len(features)))
    return np.asarray(features, dtype=np.float32).reshape(1, -1)


# ── routes ─────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status")
def status():
    """Quick check so the frontend knows the model is loaded."""
    return jsonify({"model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json(silent=True)
    if not data or "frame" not in data:
        return jsonify({"error": "No frame data"}), 400

    # ── decode base64 → OpenCV image ──
    try:
        header, encoded = data["frame"].split(",", 1)
        img_bytes = base64.b64decode(encoded)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("imdecode returned None")
    except Exception as e:
        return jsonify({"error": f"Bad frame: {e}"}), 400

    # ── MediaPipe ──
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return jsonify({
            "prediction": None,
            "confidence": 0,
            "hands_detected": 0,
        })

    # ── normalise + classify ──
    features = extract_features(result.multi_hand_landmarks)
    probabilities = model.predict_proba(features)[0]
    best_idx = int(np.argmax(probabilities))
    prediction = str(model.classes_[best_idx])
    confidence = float(probabilities[best_idx])

    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "hands_detected": len(result.multi_hand_landmarks),
    })


# ── sentence builder API ──────────────────────────────────────

@app.route("/sentence", methods=["POST"])
def sentence_api():
    """
    Actions: add, space, backspace, clear, get, set
    Body:  { "action": "add", "char": "A" }
    """
    data = request.get_json(silent=True) or {}
    action = data.get("action", "get")

    if action == "add":
        ch = data.get("char", "")
        if ch.isalpha():
            builder.add_letter(ch)
        elif ch.isdigit():
            builder.add_digit(ch)
    elif action == "word":
        builder.add_word(data.get("word", ""))
    elif action == "space":
        builder.add_space()
    elif action == "backspace":
        builder.backspace()
    elif action == "clear":
        builder.clear()
    elif action == "set":
        builder.set_text(data.get("text", ""))
    # 'get' or anything else → just return current text

    return jsonify({"text": builder.get_text()})


# ── main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    load_model()
    print("[SignBridge] Starting Flask on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)