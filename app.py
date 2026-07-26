"""Local web interface for training and running the LBPH + SVM face detector."""

from __future__ import annotations

import base64
import queue
import subprocess
import sys
import threading
from pathlib import Path

import cv2
import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory

from detect_faces import classify_haar_proposals, collect_haar_proposals, load_haar_cascade, nms
from train_face_detector import GRID_SIZE, lbp_code_image, prepare_patch


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = r"D:\_ThS\CoVi\kaggle\input\face-detection-dataset"
DEFAULT_MODEL = ROOT / "models" / "lbph_svm_face_detector.joblib"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

training_process: subprocess.Popen[str] | None = None
training_log: queue.Queue[str] = queue.Queue()
training_lock = threading.Lock()


def overlay_boxes(image: np.ndarray, boxes, color, label: str = "", scores: bool = False) -> np.ndarray:
    result = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        text = label
        if scores and len(box) == 5:
            text = f"face {box[4]:.0%}"
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        if text:
            cv2.putText(result, text, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return result


def lbph_histogram_visual(code: np.ndarray) -> np.ndarray:
    """Render one normalized LBP histogram per LBPH grid cell."""
    height, width = 320, 520
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    cell_height, cell_width = height // GRID_SIZE, width // GRID_SIZE
    for row_index, row in enumerate(np.array_split(code, GRID_SIZE, axis=0)):
        for column_index, cell in enumerate(np.array_split(row, GRID_SIZE, axis=1)):
            histogram = np.bincount(cell.ravel(), minlength=256).astype(np.float32)
            histogram /= histogram.max() + 1e-7
            x0, y0 = column_index * cell_width, row_index * cell_height
            cv2.rectangle(canvas, (x0, y0), (x0 + cell_width - 1, y0 + cell_height - 1), (100, 100, 100), 1)
            baseline = y0 + cell_height - 22
            points = []
            for index, value in enumerate(histogram):
                x = x0 + 4 + int(index * (cell_width - 8) / 255)
                y = baseline - int(value * (cell_height - 38))
                points.append((x, y))
            cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, (220, 80, 20), 1)
            cv2.putText(canvas, f"Grid ({row_index + 1},{column_index + 1})", (x0 + 6, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1)
    return canvas


def image_data_url(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise RuntimeError("Không thể chuyển ảnh kết quả để hiển thị.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def run_detection(image: np.ndarray, model_path: Path, threshold: float, min_size: int) -> tuple[dict[str, str], int]:
    model = joblib.load(model_path)
    if model.get("score_type") != "face_probability":
        raise RuntimeError("Model cũ chưa được calibration. Hãy train lại bằng ứng dụng này.")
    cascade = load_haar_cascade("haarcascade_frontalface_default.xml")
    proposals = collect_haar_proposals(image, cascade, min_size)
    accepted = classify_haar_proposals(image, model["classifier"], proposals, threshold)
    final = nms(accepted, 0.30)

    source_box = (final or accepted or [(0, 0, image.shape[1], image.shape[0], 0.0)])[0]
    x1, y1, x2, y2 = source_box[:4]
    patch = image[y1:y2, x1:x2]
    normalized = prepare_patch(patch)
    code = lbp_code_image(patch)
    stages = {
        "original": image,
        "gray": cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        "proposals": overlay_boxes(image, proposals, (0, 165, 255), "proposal"),
        "patch": patch,
        "normalized": normalized,
        "lbp": cv2.resize(code, (256, 256), interpolation=cv2.INTER_NEAREST),
        "histogram": lbph_histogram_visual(code),
        "svm": overlay_boxes(image, accepted, (0, 255, 255), scores=True),
        "final": overlay_boxes(image, final, (0, 255, 0), scores=True),
    }
    return {name: image_data_url(stage) for name, stage in stages.items()}, len(final)


def resolve_model_path(value: str) -> Path:
    path = Path(value).expanduser() if value else DEFAULT_MODEL
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise RuntimeError(f"Không tìm thấy model: {path}")
    return path


def stream_training_output(process: subprocess.Popen[str]) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        training_log.put(line)
    training_log.put(f"\n--- Kết thúc train (exit code {process.wait()}) ---\n")


@app.get("/")
def index():
    return render_template("index.html", default_model=str(DEFAULT_MODEL), default_dataset=DEFAULT_DATASET)


@app.get("/images.png")
def school_logo():
    return send_from_directory(ROOT / "templates", "images.png")


@app.get("/api/health")
def health():
    return jsonify(status="ok", model_exists=DEFAULT_MODEL.is_file())


@app.post("/api/detect")
def detect_api():
    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return jsonify(error="Hãy chọn ảnh đầu vào."), 400
    try:
        threshold = float(request.form.get("threshold", "0.41"))
        min_size = int(request.form.get("min_size", "40"))
        if not 0 <= threshold <= 1 or min_size < 1:
            raise ValueError
        encoded = np.frombuffer(uploaded.read(), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Không thể đọc ảnh đã tải lên.")
        model_path = resolve_model_path(request.form.get("model_path", ""))
        stages, count = run_detection(image, model_path, threshold, min_size)
        return jsonify(count=count, stages=stages)
    except ValueError:
        return jsonify(error="Ngưỡng phải từ 0 đến 1 và kích thước tối thiểu phải lớn hơn 0."), 400
    except Exception as error:
        return jsonify(error=str(error)), 500


@app.post("/api/train")
def train_api():
    global training_process
    with training_lock:
        if training_process is not None and training_process.poll() is None:
            return jsonify(error="Tiến trình train đang chạy."), 409
        try:
            payload = request.get_json()
            dataset = payload["dataset"].strip()
            output = payload["output"].strip()
            values = [int(payload[name]) for name in ("max_train", "max_val", "negatives", "samples")]
            if not dataset or not output or any(value < 0 for value in values):
                raise ValueError
        except (AttributeError, KeyError, TypeError, ValueError):
            return jsonify(error="Kiểm tra lại đường dẫn và các tham số train."), 400
        while not training_log.empty():
            training_log.get_nowait()
        command = [
            sys.executable, "-u", str(ROOT / "train_face_detector.py"), "--dataset", dataset,
            "--output", output, "--max-train-images", str(values[0]), "--max-val-images", str(values[1]),
            "--negatives-per-image", str(values[2]), "--max-samples-per-class", str(values[3]),
        ]
        training_process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        threading.Thread(target=stream_training_output, args=(training_process,), daemon=True).start()
    return jsonify(status="started")


@app.get("/api/training")
def training_api():
    lines = []
    while not training_log.empty():
        lines.append(training_log.get_nowait())
    running = training_process is not None and training_process.poll() is None
    return jsonify(running=running, log="".join(lines))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)