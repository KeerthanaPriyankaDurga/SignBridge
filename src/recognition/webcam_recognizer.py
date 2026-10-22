
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "alphabet_only_classifier.joblib"


def normalize_hand(hand):
    wrist = hand.landmark[0]
    points = []

    for point in hand.landmark:
        points.extend([
            point.x - wrist.x,
            point.y - wrist.y,
            point.z - wrist.z,
        ])

    scale = max(abs(value) for value in points)

    if scale > 0:
        points = [value / scale for value in points]

    return points


def extract_features(hand_landmarks):
    features = []

    for hand in hand_landmarks[:2]:
        features.extend(normalize_hand(hand))

    features = features[:126]
    features.extend([0.0] * (126 - len(features)))

    return np.asarray(features, dtype=np.float32).reshape(1, -1)


def main():
    if not MODEL_PATH.exists():
        print(f"Alphabet model not found: {MODEL_PATH}")
        return

    model = joblib.load(MODEL_PATH)
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    drawing = mp.solutions.drawing_utils
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Camera could not be opened.")
        hands.close()
        return

    print("SIGN BRIDGE - Alphabet Recognition")
    print("Show one letter sign. Press Q to quit.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            label = "Show one hand sign"

            if result.multi_hand_landmarks:
                features = extract_features(result.multi_hand_landmarks)
                probabilities = model.predict_proba(features)[0]
                best_index = int(np.argmax(probabilities))
                prediction = model.classes_[best_index]
                confidence = float(probabilities[best_index])

                label = f"{prediction} ({confidence * 100:.1f}%)"

                for hand in result.multi_hand_landmarks:
                    drawing.draw_landmarks(
                        frame,
                        hand,
                        mp.solutions.hands.HAND_CONNECTIONS,
                    )

            cv2.putText(
                frame,
                label,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("SIGN BRIDGE - Alphabet Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
