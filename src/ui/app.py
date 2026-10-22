"""
SignBridge Flask backend.

Serves the frontend and exposes endpoints:
  /predict       — single-frame SVM classification (letters + digits)
  /word_record   — start/stop/frame for LSTM word-sign recognition
  /sentence      — sentence builder actions
  /status        — model status check
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

# Let Python find project packages
sys.path.insert(0, str(ROOT))
from src.sentence_builder.builder import SentenceBuilder
from src.recognition.word_recognizer import WordRecognizer

# ── Flask app ──────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "templates"),
)

# ── globals loaded once on startup ─────────────────────────────
model = None
hands = None
builder = SentenceBuilder()
word_rec = WordRecognizer()


def load_model():
    global model, hands
    if not MODEL_PATH.exists():
        print(f"[SignBridge] SVM model not found: {MODEL_PATH}")
        print("  Run train_classifier.py first.")
        return False
    model = joblib.load(MODEL_PATH)
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    print(f"[SignBridge] SVM loaded — classes: {list(model.classes_)}")
    return True


def load_word_model():
    """Try to load the LSTM word model (optional — project works without it)."""
    ok = word_rec.load()
    if not ok:
        print("[SignBridge] Word LSTM not found — word mode disabled.")
        print("  Run train_word_lstm.py to enable it.")
    return ok


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


def decode_frame(data):
    """Decode a base64 video frame from the browser into an OpenCV image."""
    header, encoded = data["frame"].split(",", 1)
    img_bytes = base64.b64decode(encoded)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("imdecode returned None")
    return frame


# ── routes ─────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status")
def status():
    """Frontend checks which models are available."""
    return jsonify({
        "model_loaded": model is not None,
        "word_model_loaded": word_rec.model is not None,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Single-frame SVM prediction for letters/digits."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json(silent=True)
    if not data or "frame" not in data:
        return jsonify({"error": "No frame data"}), 400

    try:
        frame = decode_frame(data)
    except Exception as e:
        return jsonify({"error": f"Bad frame: {e}"}), 400

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return jsonify({
            "prediction": None,
            "confidence": 0,
            "hands_detected": 0,
        })

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


# ── word recognition (LSTM) ──────────────────────────────────

@app.route("/word_record", methods=["POST"])
def word_record():
    """
    Word-sign recording endpoint.  The frontend sends:
      { "action": "start" }           — begin buffering frames
      { "action": "frame", "frame": "data:image/jpeg;base64,..." }
                                       — add one frame to buffer
      { "action": "stop" }            — stop recording + predict
    """
    if word_rec.model is None:
        return jsonify({"error": "Word model not loaded"}), 503

    data = request.get_json(silent=True) or {}
    action = data.get("action", "")

    if action == "start":
        word_rec.start_recording()
        return jsonify({"status": "recording"})

    elif action == "frame":
        if "frame" not in data:
            return jsonify({"error": "No frame data"}), 400
        try:
            frame = decode_frame(data)
        except Exception as e:
            return jsonify({"error": f"Bad frame: {e}"}), 400

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        hand_data = result.multi_hand_landmarks if result.multi_hand_landmarks else None
        word_rec.add_frame(hand_data)

        return jsonify({
            "status": "recording",
            "frames": len(word_rec.buffer),
        })

    elif action == "stop":
        result = word_rec.stop_recording()
        if result is None:
            return jsonify({
                "prediction": None,
                "confidence": 0,
                "error": "Too few frames — hold longer",
            })
        word, confidence = result
        return jsonify({
            "prediction": word,
            "confidence": round(confidence, 4),
        })

    return jsonify({"error": f"Unknown action: {action}"}), 400


# ── sentence builder API ──────────────────────────────────────

@app.route("/sentence", methods=["POST"])
def sentence_api():
    """
    Actions: add, space, backspace, clear, get, set, word
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

    return jsonify({"text": builder.get_text()})


# ── main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    load_model()
    load_word_model()
    print("[SignBridge] Starting Flask on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)