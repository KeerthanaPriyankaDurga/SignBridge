"""
Extract landmark sequences from ISL word-sign videos.

For each video:
  1. Read every frame with OpenCV
  2. Run MediaPipe Hands to get landmarks
  3. Normalise each frame's landmarks (wrist-relative + scale)
     — same normalize_hand() used in the static pipeline
  4. Pad or truncate all sequences to a fixed length (SEQ_LEN)
  5. Save as .npy arrays ready for LSTM training

Usage:
    python src/data/extract_word_features.py "C:/path/to/word_dataset"

The dataset folder should contain sub-folders named like:
    9.hello/   38.thankyou/   32.sorry/   ...
Each sub-folder holds .mp4 (or .avi/.mov) video files.
"""

import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# ── config ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "word_sequences"

# Fixed sequence length — all sequences are padded/truncated to this.
# 30 frames ≈ 1 second of video at 30fps, which covers most short
# ISL word signs.  Adjust if your videos are longer.
SEQ_LEN = 30

# Which words to extract (must match the folder name AFTER the number
# prefix, e.g. "9.hello" → "hello").
WORDS = [
    "hello", "thankyou", "sorry", "stop", "help",
    "good", "bad", "eat", "drinking", "welcome",
]

# Features per frame: 126 (2 hands × 21 landmarks × 3 coords)
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


# ── video processing ─────────────────────────────────────────────

def extract_sequence_from_video(video_path, hands):
    """
    Read a video file, extract normalised landmark features per frame.
    Returns a list of 126-float vectors (one per frame where hands
    were detected).  Frames with no hands are filled with zeros so
    the temporal position is preserved.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Could not open: {video_path.name}")
        return None

    sequence = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            features = extract_frame_features(result.multi_hand_landmarks)
        else:
            # No hands detected in this frame — keep a zero vector so
            # the LSTM still sees the correct temporal position.
            features = [0.0] * FEATURES_PER_FRAME

        sequence.append(features)

    cap.release()

    if len(sequence) == 0:
        print(f"  No frames read: {video_path.name}")
        return None

    return sequence


def pad_or_truncate(sequence, target_len):
    """
    Make every sequence exactly `target_len` frames long.
    - Too long  → take the MIDDLE portion (the sign is usually
      centred in the video, with idle start/end).
    - Too short → pad with zeros at the end.
    """
    arr = np.array(sequence, dtype=np.float32)

    if len(arr) > target_len:
        # Centre-crop
        start = (len(arr) - target_len) // 2
        arr = arr[start : start + target_len]
    elif len(arr) < target_len:
        pad = np.zeros((target_len - len(arr), FEATURES_PER_FRAME),
                       dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)

    return arr  # shape: (target_len, 126)


# ── main ─────────────────────────────────────────────────────────

def find_word_folder(dataset_root, word):
    """
    Find the folder for a word.  Handles two naming conventions:
      - Numbered:  '9.hello', '38.thankyou'  (split on first dot)
      - Plain:     'hello', 'drinking'       (exact match)
    """
    for folder in dataset_root.iterdir():
        if not folder.is_dir():
            continue
        name = folder.name.lower()
        # Plain match: folder is just the word name
        if name == word.lower():
            return folder
        # Numbered prefix: "9.hello" → match "hello"
        parts = folder.name.split(".", 1)
        if len(parts) == 2 and parts[1].lower() == word.lower():
            return folder
    return None


def main():
    if len(sys.argv) != 2:
        print('Usage: python src/data/extract_word_features.py '
              '"C:/path/to/word_dataset"')
        return

    dataset_root = Path(sys.argv[1])
    if not dataset_root.is_dir():
        print(f"Dataset folder not found: {dataset_root}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_sequences = []  # list of (sequence_array, label)
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    with mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        for word in WORDS:
            folder = find_word_folder(dataset_root, word)
            if folder is None:
                print(f"[SKIP] No folder found for '{word}'")
                continue

            videos = sorted(
                p for p in folder.iterdir()
                if p.suffix.lower() in video_exts
            )
            print(f"\n[{word.upper()}] Found {len(videos)} videos "
                  f"in {folder.name}/")

            for vid in videos:
                seq = extract_sequence_from_video(vid, hands)
                if seq is None:
                    continue

                padded = pad_or_truncate(seq, SEQ_LEN)
                all_sequences.append((padded, word.upper()))
                print(f"  {vid.name}: {len(seq)} frames → "
                      f"{SEQ_LEN} (padded/cropped)")

    if not all_sequences:
        print("\nNo sequences extracted. Check your dataset path.")
        return

    # Build numpy arrays
    X = np.array([s[0] for s in all_sequences], dtype=np.float32)
    y = np.array([s[1] for s in all_sequences])

    # Save
    np.save(OUTPUT_DIR / "word_X.npy", X)
    np.save(OUTPUT_DIR / "word_y.npy", y)

    labels, counts = np.unique(y, return_counts=True)
    print(f"\n{'='*50}")
    print(f"Total sequences: {len(X)}")
    print(f"Sequence shape:  {X.shape}  "
          f"(samples, {SEQ_LEN} frames, {FEATURES_PER_FRAME} features)")
    print(f"Classes: {len(labels)}")
    for label, count in zip(labels, counts):
        print(f"  {label}: {count} samples")
    print(f"\nSaved to: {OUTPUT_DIR}")
    print(f"  word_X.npy  — features array")
    print(f"  word_y.npy  — labels array")


if __name__ == "__main__":
    main()