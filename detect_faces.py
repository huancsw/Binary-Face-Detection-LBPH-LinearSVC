"""Run the trained LBPH-style SVM detector on a still image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np

from train_face_detector import PATCH_SIZE, lbph_feature


def load_haar_cascade(filename: str) -> cv2.CascadeClassifier:
    """Load an OpenCV 4.x Haar cascade with a clear error on OpenCV 5.x."""
    cascade_factory = getattr(cv2, "CascadeClassifier", None)
    if cascade_factory is None:
        raise RuntimeError(
            "Haar Cascade requires OpenCV 4.x. Reinstall dependencies with: "
            "python -m pip install --upgrade --force-reinstall 'opencv-python>=4.10,<5'"
        )
    cascade = cascade_factory(cv2.data.haarcascades + filename)
    if cascade.empty():
        raise RuntimeError(f"Cannot load Haar cascade: {filename}")
    return cascade


def nms(boxes: list[tuple[int, int, int, int, float]], threshold: float) -> list[tuple[int, int, int, int, float]]:
    boxes = sorted(boxes, key=lambda item: item[4], reverse=True)
    kept = []
    while boxes:
        best = boxes.pop(0)
        kept.append(best)
        remaining = []
        for box in boxes:
            xx1, yy1 = max(best[0], box[0]), max(best[1], box[1])
            xx2, yy2 = min(best[2], box[2]), min(best[3], box[3])
            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            union = (best[2] - best[0]) * (best[3] - best[1]) + (box[2] - box[0]) * (box[3] - box[1]) - inter
            if inter / union < threshold:
                remaining.append(box)
        boxes = remaining
    return kept


def detect(image: np.ndarray, classifier, probability_threshold: float, stride: int) -> list[tuple[int, int, int, int, float]]:
    detections = []
    height, width = image.shape[:2]
    for scale in (1.0, 0.85, 0.70, 0.55, 0.40, 0.28):
        scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        for y in range(0, scaled.shape[0] - PATCH_SIZE + 1, stride):
            for x in range(0, scaled.shape[1] - PATCH_SIZE + 1, stride):
                probability = classifier.predict_proba([lbph_feature(scaled[y:y + PATCH_SIZE, x:x + PATCH_SIZE])])[0, 1]
                if probability >= probability_threshold:
                    detections.append((int(x / scale), int(y / scale), int((x + PATCH_SIZE) / scale), int((y + PATCH_SIZE) / scale), float(probability)))
    return nms(detections, 0.35)


def collect_haar_proposals(
    image: np.ndarray, cascade: cv2.CascadeClassifier, min_face_size: int
) -> list[tuple[int, int, int, int]]:
    """Return frontal Haar candidate boxes."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_height, image_width = gray.shape
    proposals = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_face_size, min_face_size)
    )

    boxes = []
    for x, y, width, height in proposals:
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(image_width, int(x + width)), min(image_height, int(y + height))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def classify_haar_proposals(
    image: np.ndarray, classifier, proposals: list[tuple[int, int, int, int]], probability_threshold: float
) -> list[tuple[int, int, int, int, float]]:
    """Classify Haar proposals with the LBPH + SVM model, before NMS."""
    detections = []
    for x1, y1, x2, y2 in proposals:
        probability = classifier.predict_proba([lbph_feature(image[y1:y2, x1:x2])])[0, 1]
        if probability >= probability_threshold:
            detections.append((x1, y1, x2, y2, float(probability)))
    return detections


def detect_haar_proposals(
    image: np.ndarray, classifier, cascade: cv2.CascadeClassifier, probability_threshold: float, min_face_size: int
) -> list[tuple[int, int, int, int, float]]:
    """Use Haar only for proposals; LBPH + SVM makes the final decision."""
    proposals = collect_haar_proposals(image, cascade, min_face_size)
    return nms(classify_haar_proposals(image, classifier, proposals, probability_threshold), 0.30)

def main() -> None:
    parser = argparse.ArgumentParser(description="Detect faces using the trained LBPH-style SVM model.")
    parser.add_argument("--model", type=Path, default=Path("models/lbph_svm_face_detector.joblib"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("detections.jpg"))
    parser.add_argument("--probability-threshold", type=float, default=0.41)
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=40,
        help="Minimum face proposal size in pixels; lower this only to detect very small faces.",
    )
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument(
        "--proposal",
        choices=("haar", "sliding"),
        default="haar",
        help="'haar' gives face-sized proposals then verifies them with SVM; 'sliding' is slower and less precise.",
    )
    args = parser.parse_args()

    model = joblib.load(args.model)
    if model.get("score_type") != "face_probability":
        raise SystemExit("This is an old uncalibrated model. Retrain it with train_face_detector.py before detection.")
    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    if args.proposal == "haar":
        cascade = load_haar_cascade("haarcascade_frontalface_default.xml")
        detections = detect_haar_proposals(
            image, model["classifier"], cascade, args.probability_threshold, args.min_face_size
        )
    else:
        detections = detect(image, model["classifier"], args.probability_threshold, args.stride)
    for x1, y1, x2, y2, score in detections:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"face {score:.0%}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), image)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
