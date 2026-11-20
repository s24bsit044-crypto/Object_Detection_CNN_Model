"""Mini Object Detection - Streamlit app (YOLO + OpenCV motion detection)."""
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.detectors import (
    MotionDetector,
    detect_with_yolo,
    load_yolo,
    read_video_frames,
)

ROOT = Path(__file__).parent
SAMPLES = ROOT / "samples"

st.set_page_config(page_title="Mini Object Detection", page_icon="🎯", layout="wide")


@st.cache_resource(show_spinner="Loading YOLO model (first run downloads weights)...")
def get_model():
    return load_yolo()


def to_bgr(pil_image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)


def to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def png_bytes(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    return buf.tobytes() if ok else b""


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🎯 Settings")
conf = st.sidebar.slider("YOLO confidence threshold", 0.10, 0.95, 0.50, 0.05)
min_area = st.sidebar.slider("Motion: minimum object area (px)", 200, 10000, 1500, 100)
st.sidebar.caption("Built with Ultralytics YOLO, OpenCV and Streamlit.")

st.title("🎯 Mini Object Detection")
st.write("Detect objects with **YOLO** and moving objects with **OpenCV** "
         "(background subtraction + contours).")

tab_yolo, tab_motion, tab_about = st.tabs(
    ["🧠 YOLO Object Detection", "🏃 Motion Detection", "ℹ️ About"]
)

# ------------------------------------------------------------------ YOLO tab
with tab_yolo:
    source = st.radio("Image source", ["Sample image", "Upload image", "Camera"],
                      horizontal=True)
    image_bgr = None

    if source == "Sample image":
        choice = st.selectbox("Choose a sample", sorted(p.name for p in SAMPLES.glob("*.jpg")))
        image_bgr = cv2.imread(str(SAMPLES / choice))
    elif source == "Upload image":
        up = st.file_uploader("Upload a JPG/PNG image", type=["jpg", "jpeg", "png"])
        if up:
            image_bgr = to_bgr(Image.open(up))
    else:
        cam = st.camera_input("Take a picture (allow camera access in your browser)")
        if cam:
            image_bgr = to_bgr(Image.open(cam))

    if image_bgr is not None:
        if st.button("Run YOLO detection", type="primary"):
            with st.spinner("Detecting..."):
                annotated, detections = detect_with_yolo(get_model(), image_bgr, conf)
            c1, c2 = st.columns(2)
            c1.image(to_rgb(image_bgr), caption="Input", width="stretch")
            c2.image(to_rgb(annotated), caption=f"Detected: {len(detections)} object(s)",
                     width="stretch")
            if detections:
                st.dataframe(pd.DataFrame(detections), width="stretch")
                st.bar_chart(pd.DataFrame(detections)["class"].value_counts())
            else:
                st.info("No objects found. Try lowering the confidence threshold.")
            st.download_button("Download result image", png_bytes(annotated),
                               "detection_result.png", "image/png")
        else:
            st.image(to_rgb(image_bgr), caption="Input", width=500)

# ------------------------------------------------------------------ Motion tab
with tab_motion:
    st.write("The first frames teach the detector what the background looks like, "
             "so objects that *enter* the scene are detected.")
    vsrc = st.radio("Video source", ["Sample video", "Upload video"], horizontal=True)
    video_path = None

    if vsrc == "Sample video":
        video_path = str(SAMPLES / "sample_motion.mp4")
    else:
        vid = st.file_uploader("Upload a short video (mp4/avi/mov)",
                               type=["mp4", "avi", "mov", "mkv"])
        if vid:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(vid.name).suffix)
            tmp.write(vid.read())
            tmp.close()
            video_path = tmp.name

    if video_path and st.button("Run motion detection", type="primary"):
        frames = read_video_frames(video_path, max_frames=150, step=2)
        if not frames:
            st.error("Could not read any frames from this video.")
        else:
            detector = MotionDetector(min_area=min_area)
            warmup = min(10, len(frames) // 4)
            progress = st.progress(0.0)
            results = []
            for i, f in enumerate(frames):
                results.append(detector.process(f))
                progress.progress((i + 1) / len(frames))
            progress.empty()

            counts = [r[2] for r in results]
            hits = [i for i in range(warmup, len(results)) if counts[i] > 0]
            st.success(f"Processed {len(frames)} frames. "
                       f"Motion found in {len(hits)} frame(s).")
            st.line_chart(pd.DataFrame({"moving objects": counts}))

            if hits:
                step = max(1, len(hits) // 3)
                for i in hits[::step][:3]:
                    annotated, mask, count = results[i]
                    a, b = st.columns(2)
                    a.image(to_rgb(annotated), caption=f"Frame {i}: {count} object(s)",
                            width="stretch")
                    b.image(mask, caption="Cleaned motion mask", width="stretch")
            else:
                st.info("No motion detected. Lower the minimum area or use a different video.")

# ------------------------------------------------------------------ About tab
with tab_about:
    st.markdown("""
### How it works
- **YOLO**: a pretrained Ultralytics nano model (trained on the COCO dataset,
  80 classes such as person, bus, bottle, phone) detects objects in an image.
- **Motion detection**: OpenCV `MOG2` background subtraction, morphology to clean
  noise, then contours filtered by area to count moving objects.

### Notes
- No dataset needs to be downloaded; the YOLO model is already trained on COCO.
- On the free Streamlit Cloud (CPU only) the first detection is slower because the
  model weights are downloaded once.
""")
