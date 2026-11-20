"""Core detection logic (no Streamlit code here, so it is easy to test).

Two detectors, taken from the original Colab notebook:
1. MotionDetector  - classic OpenCV pipeline (MOG2 + morphology + contours)
2. YOLO detection  - pretrained Ultralytics model (COCO, 80 classes)
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_CANDIDATES = ("yolo26n.pt", "yolo11n.pt")


# --------------------------------------------------------------------------
# 1. Motion detection (OpenCV)
# --------------------------------------------------------------------------
class MotionDetector:
    """Background subtraction + contour filtering."""

    def __init__(self, min_area: int = 1500):
        self.min_area = min_area
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """Return (annotated_frame, clean_mask, moving_object_count)."""
        mask = self.bg.apply(frame)
        _, binary = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel)
        clean = cv2.dilate(clean, self.kernel, iterations=2)

        contours, _ = cv2.findContours(
            clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        annotated = frame.copy()
        count = 0
        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            count += 1
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(annotated, f"Object {count}", (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated, f"Moving objects: {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        return annotated, clean, count


def detect_moving_objects(frames: List[np.ndarray], min_area: int = 1500):
    """Run the motion pipeline over a list of BGR frames."""
    detector = MotionDetector(min_area=min_area)
    return [detector.process(f) for f in frames]


def read_video_frames(path: str, max_frames: int = 150, step: int = 2,
                      max_width: int = 640) -> List[np.ndarray]:
    """Read up to `max_frames` frames (every `step`-th) and resize them."""
    cap = cv2.VideoCapture(path)
    frames: List[np.ndarray] = []
    index = 0
    try:
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step == 0:
                h, w = frame.shape[:2]
                if w > max_width:
                    frame = cv2.resize(frame, (max_width, int(h * max_width / w)))
                frames.append(frame)
            index += 1
    finally:
        cap.release()
    return frames


# --------------------------------------------------------------------------
# 2. YOLO detection
# --------------------------------------------------------------------------
def load_yolo():
    """Load YOLO26 nano; fall back to YOLO11 nano (same as the notebook)."""
    from ultralytics import YOLO

    MODELS_DIR.mkdir(exist_ok=True)
    last_error = None
    for name in MODEL_CANDIDATES:
        local = MODELS_DIR / name
        try:
            return YOLO(str(local) if local.exists() else name)
        except Exception as e:  # noqa: BLE001
            last_error = e
    raise RuntimeError(f"No YOLO model could be loaded: {last_error}")


def detect_with_yolo(model, frame: np.ndarray, conf: float = 0.50):
    """Return (annotated_bgr_frame, list_of_detection_dicts)."""
    result = model.predict(source=frame, conf=conf, verbose=False)[0]
    annotated = result.plot()
    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        detections.append({
            "class": result.names[class_id],
            "confidence": round(float(box.conf[0]), 3),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })
    cv2.putText(annotated, f"Objects: {len(detections)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    return annotated, detections
