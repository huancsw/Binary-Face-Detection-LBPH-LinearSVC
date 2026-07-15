"""Train an LBPH-style + SVM binary face detector from YOLO annotations.

The Face Detection Dataset has one class (``Human face``) and YOLO labels.
This script turns annotated face boxes into positive patches and samples patches
that do not overlap any annotation as negatives.  It then uses local binary
pattern histograms as features and trains a linear SVM with labels:

    1 = face, 0 = background / not a face

No identity label and no Haar cascade is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC


PATCH_SIZE = 64
# A 2x2 grid gives 1,024 features per patch.  The original 4x4 grid produced
# 4,096 features and exhausts memory when all annotated faces are loaded.
GRID_SIZE = 2
RANDOM_SEED = 42


def read_yolo_boxes(label_path: Path, width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Read class-0 YOLO boxes and return clipped (x1, y1, x2, y2) boxes."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 5 or values[0] != "0":
            continue
        _, cx, cy, bw, bh = map(float, values)
        x1 = max(0, int(round((cx - bw / 2) * width)))
        y1 = max(0, int(round((cy - bh / 2) * height)))
        x2 = min(width, int(round((cx + bw / 2) * width)))
        y2 = min(height, int(round((cy + bh / 2) * height)))
        if x2 - x1 >= 8 and y2 - y1 >= 8:
            boxes.append((x1, y1, x2, y2))
    return boxes


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if not intersection:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def prepare_patch(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.resize(gray, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(gray)


def lbph_feature(image: np.ndarray) -> np.ndarray:
    """Return an LBP histogram per spatial cell (the feature used by LBPH)."""
    gray = prepare_patch(image)
    center = gray[1:-1, 1:-1]
    code = np.zeros_like(center, dtype=np.uint8)
    neighbors = (
        gray[:-2, :-2], gray[:-2, 1:-1], gray[:-2, 2:], gray[1:-1, 2:],
        gray[2:, 2:], gray[2:, 1:-1], gray[2:, :-2], gray[1:-1, :-2],
    )
    for bit, neighbor in enumerate(neighbors):
        code |= ((neighbor >= center).astype(np.uint8) << bit)

    rows = np.array_split(code, GRID_SIZE, axis=0)
    histograms = []
    for row in rows:
        for cell in np.array_split(row, GRID_SIZE, axis=1):
            histogram = np.bincount(cell.ravel(), minlength=256).astype(np.float32)
            histogram /= histogram.sum() + 1e-7
            histograms.append(np.sqrt(histogram))  # Hellinger normalization
    return np.concatenate(histograms)


def random_negative_boxes(
    width: int, height: int, faces: list[tuple[int, int, int, int]], count: int, rng: np.random.Generator
) -> list[tuple[int, int, int, int]]:
    """Sample background windows with IoU < 0.05 against every face box."""
    boxes, attempts = [], 0
    while len(boxes) < count and attempts < count * 60:
        attempts += 1
        size = int(rng.integers(max(24, min(width, height) // 12), max(25, min(width, height) // 2)))
        if size >= width or size >= height:
            continue
        x1, y1 = int(rng.integers(0, width - size)), int(rng.integers(0, height - size))
        candidate = (x1, y1, x1 + size, y1 + size)
        if all(iou(candidate, face) < 0.05 for face in faces):
            boxes.append(candidate)
    return boxes


def make_samples(
    dataset: Path,
    split: str,
    max_images: int,
    negatives_per_image: int,
    max_samples_per_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a balanced, bounded sample set using reservoir sampling.

    Every source image is still visited.  Once a class reaches its limit, a
    newly encountered patch randomly replaces an older one, so the retained
    set is not biased toward files at the start of the directory.
    """
    image_dir, label_dir = dataset / "images" / split, dataset / "labels" / split
    image_paths = sorted(image_dir.glob("*.jpg"))
    if max_images:
        image_paths = image_paths[:max_images]
    rng = np.random.default_rng(RANDOM_SEED if split == "train" else RANDOM_SEED + 1)
    face_features: list[np.ndarray] = []
    background_features: list[np.ndarray] = []
    face_seen = background_seen = 0

    def retain(feature: np.ndarray, label: int) -> None:
        nonlocal face_seen, background_seen
        samples = face_features if label else background_features
        if label:
            face_seen += 1
            seen = face_seen
        else:
            background_seen += 1
            seen = background_seen
        if max_samples_per_class == 0 or len(samples) < max_samples_per_class:
            samples.append(feature)
        else:
            replacement = int(rng.integers(seen))
            if replacement < max_samples_per_class:
                samples[replacement] = feature

    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        faces = read_yolo_boxes(label_dir / f"{image_path.stem}.txt", width, height)
        for x1, y1, x2, y2 in faces:
            retain(lbph_feature(image[y1:y2, x1:x2]), 1)
        for x1, y1, x2, y2 in random_negative_boxes(width, height, faces, negatives_per_image, rng):
            retain(lbph_feature(image[y1:y2, x1:x2]), 0)
        if index % 1000 == 0:
            print(f"{split}: processed {index}/{len(image_paths)} images")

    if not face_features or not background_features:
        raise RuntimeError(f"No usable samples found in {image_dir}")
    features = np.asarray(face_features + background_features, dtype=np.float32)
    labels = np.asarray([1] * len(face_features) + [0] * len(background_features), dtype=np.int8)
    shuffle = rng.permutation(len(labels))
    return features[shuffle], labels[shuffle]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a binary LBPH-style SVM face detector.")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset root containing images/ and labels/.")
    parser.add_argument("--output", type=Path, default=Path("models/lbph_svm_face_detector.joblib"))
    parser.add_argument("--max-train-images", type=int, default=0, help="0 uses all 13,386 train images.")
    parser.add_argument("--max-val-images", type=int, default=0, help="0 uses all validation images.")
    parser.add_argument(
        "--negatives-per-image",
        type=int,
        default=6,
        help="Background patches sampled per image; more diverse negatives reduce false positives.",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=12000,
        help="Maximum retained face and background patches per split; 0 keeps all (requires substantially more RAM).",
    )
    args = parser.parse_args()

    for required in (args.dataset / "images" / "train", args.dataset / "labels" / "train"):
        if not required.is_dir():
            raise SystemExit(f"Missing dataset directory: {required}")

    x_train, y_train = make_samples(
        args.dataset, "train", args.max_train_images, args.negatives_per_image, args.max_samples_per_class
    )
    x_val, y_val = make_samples(
        args.dataset, "val", args.max_val_images, args.negatives_per_image, args.max_samples_per_class
    )
    print(f"Training samples: {len(y_train)} (face={y_train.sum()}, background={(y_train == 0).sum()})")
    print(f"Validation samples: {len(y_val)} (face={y_val.sum()}, background={(y_val == 0).sum()})")

    # Every LBP cell histogram is already normalized in ``lbph_feature``;
    # omitting StandardScaler avoids another full-size RAM allocation.  Sigmoid
    # calibration gives a meaningful probability threshold during detection.
    classifier = CalibratedClassifierCV(
        LinearSVC(class_weight="balanced", C=1.0, dual=False, max_iter=10000, random_state=RANDOM_SEED),
        method="sigmoid",
        cv=3,
    )
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_val)
    print(classification_report(y_val, predictions, target_names=["background", "face"], digits=4))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_val, predictions))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"classifier": classifier, "patch_size": PATCH_SIZE, "grid_size": GRID_SIZE, "score_type": "face_probability"},
        args.output,
    )
    report_path = args.output.with_suffix(".metrics.json")
    report_path.write_text(json.dumps(classification_report(y_val, predictions, output_dict=True), indent=2), encoding="utf-8")
    print(f"Saved model: {args.output}")
    print(f"Saved metrics: {report_path}")


if __name__ == "__main__":
    main()
