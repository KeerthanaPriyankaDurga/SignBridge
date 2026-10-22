
from pathlib import Path
from collections import Counter
import hashlib
import csv
import sys

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATA_EXT = VIDEO_EXT | IMAGE_EXT


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(root):
    root = Path(root).expanduser()

    if not root.exists() or not root.is_dir():
        print(f"ERROR: Invalid dataset folder: {root}")
        return

    files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in DATA_EXT
    )

    if not files:
        print("No supported image or video files found.")
        print("Check whether the correct dataset folder was selected.")
        return

    extension_counts = Counter(p.suffix.lower() for p in files)
    class_counts = Counter()
    class_details = Counter()
    total_bytes = 0
    hashes = {}
    duplicates = []

    for path in files:
        relative = path.relative_to(root)
        parts = relative.parts

        # Uses the first directory below the selected dataset root.
        class_name = parts[0] if len(parts) > 1 else "(root files)"
        ext = path.suffix.lower()
        kind = "video" if ext in VIDEO_EXT else "image"

        class_counts[class_name] += 1
        class_details[(class_name, kind)] += 1
        total_bytes += path.stat().st_size

        try:
            digest = file_hash(path)
            if digest in hashes:
                duplicates.append(
                    (str(path.relative_to(root)), hashes[digest])
                )
            else:
                hashes[digest] = str(path.relative_to(root))
        except OSError as error:
            print(f"WARNING: Could not read {path}: {error}")

    print(f"\nDataset: {root.resolve()}")
    print(f"Total supported files: {len(files)}")
    print(f"Total videos: {sum(extension_counts[e] for e in VIDEO_EXT)}")
    print(f"Total images: {sum(extension_counts[e] for e in IMAGE_EXT)}")
    print(f"Total size: {total_bytes / (1024 ** 3):.2f} GB")
    print(f"Top-level folders with data: {len(class_counts)}")

    print("\nFile extensions:")
    for ext, count in sorted(extension_counts.items()):
        print(f"  {ext}: {count}")

    print("\nTop-level folder counts:")
    for name in sorted(class_counts):
        videos = class_details[(name, "video")]
        images = class_details[(name, "image")]
        print(f"  {name}: {videos} videos, {images} images")

    print(f"\nExact duplicate files: {len(duplicates)}")
    for duplicate, original in duplicates[:20]:
        print(f"  DUPLICATE: {duplicate} == {original}")
    if len(duplicates) > 20:
        print(f"  ... and {len(duplicates) - 20} more")

    report_path = root / "dataset_audit_report.csv"
    try:
        with report_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "relative_path", "folder", "extension",
                "type", "size_bytes"
            ])

            for path in files:
                relative = path.relative_to(root)
                folder = (
                    relative.parts[0]
                    if len(relative.parts) > 1
                    else "(root files)"
                )
                ext = path.suffix.lower()
                writer.writerow([
                    str(relative),
                    folder,
                    ext,
                    "video" if ext in VIDEO_EXT else "image",
                    path.stat().st_size,
                ])

        print(f"\nCSV report saved: {report_path}")
    except OSError as error:
        print(f"WARNING: Could not save CSV report: {error}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            'Usage: python src\\data\\dataset_audit.py "DATASET_PATH"'
        )
    else:
        audit_dataset(sys.argv[1])
