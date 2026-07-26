# LBPH-style + SVM Face Detector

This repository now trains a **binary face detector**, not a face-identity recognizer:

- `1` = a face patch
- `0` = background / not a face patch

It supports the Kaggle Face Detection Dataset at
`D:\_ThS\CoVi\kaggle\input\face-detection-dataset`. The dataset's YOLO boxes in
`labels/{train,val}` are used to crop positive face patches; non-overlapping
background patches are generated as negatives. A spatial Local Binary Pattern
histogram (LBPH-style) is the feature extractor and a linear SVM is the binary
classifier. This means the code does not use person names, `cv2.face`, Haar
cascades during training, or delete dataset files.

## Install

```powershell
python -m pip install -r requirements.txt
```

The detector uses Haar cascade proposals, so it requires OpenCV 4.x. If an
existing environment has OpenCV 5 installed, force the compatible version:

```powershell
python -m pip install --upgrade --force-reinstall "opencv-python>=4.10,<5"
```

## Train

Train from all 13,386 train and 3,347 validation images:

```powershell
python train_face_detector.py --dataset "D:\_ThS\CoVi\kaggle\input\face-detection-dataset"
```

The model and validation metrics are saved to `models/`. To make a quick
smoke-test run, limit both splits:

```powershell
python train_face_detector.py --dataset "D:\_ThS\CoVi\kaggle\input\face-detection-dataset" --max-train-images 200 --max-val-images 100
```

By default, reservoir sampling retains at most 12,000 face patches and 12,000
background patches per split. This keeps the training data in RAM while still
sampling from every image. Raise or lower the limit with
`--max-samples-per-class`; use `0` only if the machine has enough RAM to keep
every patch.

## Detect in an image

```powershell
python detect_faces.py --model models/lbph_svm_face_detector.joblib --image "D:\path\to\photo.jpg" --output detections.jpg
```

The default detector uses frontal Haar cascade only to propose face-sized regions; the
LBPH + SVM model makes the final binary accept/reject decision. This avoids
sliding-window detections of eyes, noses, or beards as separate faces. It uses
a calibrated probability threshold of 41%; lower it only when a face is missed:

```powershell
python detect_faces.py --model models/lbph_svm_face_detector.joblib --image "D:\path\to\photo.jpg" --probability-threshold 0.50
```

Use `--proposal sliding` only to inspect the pure sliding-window baseline; it
is slower and typically produces more false positives.

The default ignores Haar proposals smaller than 40 pixels. Increase
`--min-face-size` when small objects such as buttons or jewellery become face
boxes.

The detector scans a multi-scale image pyramid and applies non-maximum
suppression. LBPH + SVM is a classical baseline; it will be materially less
robust than a modern YOLO detector, particularly for small, angled, or occluded
faces.

## Web interface

Run the complete browser-based application with:

```powershell
py -3 app.py
```

Open `http://127.0.0.1:5000` in a browser. The HCM-UTE branded interface has
two workspaces:

- **Phát hiện khuôn mặt**: upload an image, set the probability threshold and
	minimum face size, then view all nine processing stages from Haar proposals
	through the final NMS result.
- **Huấn luyện mô hình**: enter the YOLO dataset and output-model paths, set
	the sampling limits, and follow the live training log.

The interface identifies the authors as Lê Huy Huân and Trịnh Nguyễn Anh Hào.
