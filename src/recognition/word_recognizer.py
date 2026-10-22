"""
Word-sign recognizer using a trained LSTM model.

This module is used by app.py for live word-sign inference:
  1. Buffer normalised landmark frames as they arrive
  2. When enough frames are collected, pad/truncate to SEQ_LEN
  3. Feed the sequence into the LSTM
  4. Return the predicted word + confidence

It can also run standalone with a webcam for testing.
"""

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "word_lstm.keras"
LABELS_PATH = ROOT / "models" / "word_labels.npy"

SEQ_LEN = 30           # must match extract_word_features.py
FEATURES_PER_FRAME = 126


# ── landmark extraction (same as static pipeline) ───────────────

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


def extract_frame_features(hand_landmarks):
    """Up to 2 hands → 126-float feature vector for one frame."""
    features = []
    for hand in hand_landmarks[:2]:
        features.extend(normalize_hand(hand))
    features = features[:FEATURES_PER_FRAME]
    features.extend([0.0] * (FEATURES_PER_FRAME - len(features)))
    return features


# ── WordRecognizer class ─────────────────────────────────────────

class WordRecognizer:
    """
    Buffers landmark frames and runs LSTM inference when the buffer
    is full (SEQ_LEN frames).
    """

    def __init__(self):
        self.model = None
        self.labels = None
        self.buffer = []          # list of 126-float vectors
        self.recording = False    # True while user is holding record

    def load(self):
        """Load the trained LSTM model and label map."""
        if not MODEL_PATH.exists() or not LABELS_PATH.exists():
            print(f"[WordRecognizer] Model not found: {MODEL_PATH}")
            return False

        import tensorflow as tf
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.labels = list(np.load(LABELS_PATH))
        print(f"[WordRecognizer] Loaded — words: {self.labels}")
        return True

    def start_recording(self):
        """Start buffering frames."""
        self.buffer = []
        self.recording = True

    def add_frame(self, hand_landmarks):
        """
        Add one frame's landmarks to the buffer.
        Call this every frame while recording.
        Returns None while recording, or (word, confidence) when
        the buffer hits SEQ_LEN frames.
        """
        if not self.recording:
            return None

        if hand_landmarks:
            features = extract_frame_features(hand_landmarks)
        else:
            features = [0.0] * FEATURES_PER_FRAME

        self.buffer.append(features)

        # Don't auto-predict — let stop_recording handle it
        return None

    def stop_recording(self):
        """
        Stop recording and run prediction on whatever frames
        were collected.  Returns (word, confidence) or None.
        """
        self.recording = False

        if len(self.buffer) < 5:
            # Too few frames — probably accidental
            self.buffer = []
            return None

        return self._predict()

    def _predict(self):
        """Run the LSTM on the buffered sequence."""
        if self.model is None:
            return None

        seq = np.array(self.buffer, dtype=np.float32)

        # Pad or truncate to SEQ_LEN (centre-crop if too long)
        if len(seq) > SEQ_LEN:
            start = (len(seq) - SEQ_LEN) // 2
            seq = seq[start : start + SEQ_LEN]
        elif len(seq) < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - len(seq), FEATURES_PER_FRAME),
                           dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)

        # LSTM expects (1, SEQ_LEN, 126)
        inp = seq.reshape(1, SEQ_LEN, FEATURES_PER_FRAME)
        probs = self.model.predict(inp, verbose=0)[0]

        best_idx = int(np.argmax(probs))
        word = self.labels[best_idx]
        confidence = float(probs[best_idx])

        self.buffer = []
        return word, confidence


# ── standalone webcam test ───────────────────────────────────────

def main():
    """Test word recognition from webcam (press R to record, R to stop)."""
    recognizer = WordRecognizer()
    if not recognizer.load():
        return

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Camera could not be opened.")
        return

    print("\nSIGN BRIDGE — Word Recognition Test")
    print("Press R to start/stop recording a sign.")
    print("Press Q to quit.\n")

    label = "Press R to record"

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            hand_data = result.multi_hand_landmarks if result.multi_hand_landmarks else None

            if recognizer.recording:
                recognizer.add_frame(hand_data)
                label = f"RECORDING... ({len(recognizer.buffer)} frames)"
                color = (0, 0, 255)  # red
            else:
                color = (0, 255, 0)  # green

            cv2.putText(frame, label, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.imshow("SIGN BRIDGE — Word Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                if recognizer.recording:
                    result = recognizer.stop_recording()
                    if result:
                        word, conf = result
                        label = f"{word} ({conf*100:.1f}%)"
                        print(f"  Predicted: {word}  ({conf*100:.1f}%)")
                    else:
                        label = "Too short — try again"
                else:
                    recognizer.start_recording()

    finally:
        camera.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()