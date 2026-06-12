# YUBI Supervision

Minimal full-screen webcam vision app using MediaPipe, YOLOv8, and [supervision](https://github.com/roboflow/supervision) for live detection and keypoint visualization.

## Capabilities

- **Objects** — YOLOv8s with ByteTrack tracking, per-class corner boxes & labels (80 COCO classes)
- **Body pose** — MediaPipe pose skeleton (green), up to 2 people
- **Face mesh** — MediaPipe face landmarks (coral), up to 2 faces
- **Hands** — MediaPipe hand landmarks (blue), up to 2 hands

## Controls

- **Layer toggles** — show/hide objects, pose, face, or hands live
- **Sensitivity slider** — adjust object detection confidence threshold
- **Keyboard** — `Space` start · `Q` / `Esc` stop

## Quick start

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), click **Start Vision**, and press `Q` or `Esc` to stop.

On first run:

- MediaPipe `.task` models download into `backend/models/`
- YOLOv8s weights (`yolov8s.pt`) download automatically via Ultralytics

## macOS camera permission

If the webcam fails to open, grant camera access to your terminal (or Python) under **System Settings → Privacy & Security → Camera**.

## Local supervision checkout

To develop against a local clone of supervision:

```bash
pip install -e /path/to/supervision
```

Or run without installing:

```bash
PYTHONPATH=/path/to/supervision/src uvicorn backend.main:app --reload
```
