
import csv
from pathlib import Path

import cv2
import mediapipe as mp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = Path(
    r"C:\Users\Keerthanav\Downloads\Dataset\Dataset"
)
OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "alphabet_features.csv"
)

CLASSES = {
    "English": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    "Numbers": list("0123456789"),
}


def normalize_hand(hand):
    """Convert 21 hand landmarks into 63 wrist-relative features."""
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


def extract_features(image_path, hands):
    image = cv2.imread(str(image_path))

    if image is None:
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image)

    if not result.multi_hand_landmarks:
        return None

    features = []

    for hand in result.multi_hand_landmarks[:2]:
        features.extend(normalize_hand(hand))

    # Pad missing second-hand features with zeros.
    features = features[:126]
    features.extend([0.0] * (126 - len(features)))

    return features


def main():
    if not DATASET_ROOT.is_dir():
        print(f"Dataset folder not found: {DATASET_ROOT}")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
    ) as hands:
        with OUTPUT_FILE.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["label", "source", "features"])

            for group, labels in CLASSES.items():
                for label in labels:
                    class_dir = DATASET_ROOT / group / "RGB" / label

                    if not class_dir.is_dir():
                        print(f"Missing class folder: {class_dir}")
                        continue

                    for image_path in sorted(class_dir.glob("*.jpg")):
                        features = extract_features(image_path, hands)

                        if features is None:
                            skipped += 1
                            continue

                        writer.writerow([
                            label,
                            str(image_path),
                            " ".join(map(str, features)),
                        ])
                        processed += 1

                        if processed % 50 == 0:
                            print(f"Processed: {processed}")

    print("\nFeature extraction finished.")
    print(f"Successful images: {processed}")
    print(f"Skipped images: {skipped}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
