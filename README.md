# 🎯 Mini Object Detection

A Streamlit web app that combines two computer-vision techniques:

| Feature | Technique |
|---|---|
| **Object detection** | Ultralytics **YOLO** (nano model, pretrained on COCO – 80 classes) |
| **Motion detection** | **OpenCV** MOG2 background subtraction + morphology + contour filtering |

Input can be a sample image, an uploaded image, or a browser camera snapshot (YOLO),
and a sample or uploaded video (motion detection).

This project was converted from the Colab notebook in `notebooks/`.

## Project structure

```
├── app.py                  # Streamlit UI
├── src/detectors.py        # YOLO + motion detection logic
├── samples/                # bus.jpg, zidane.jpg, sample_motion.mp4
├── models/                 # YOLO weights (yolo26n.pt, yolo11n.pt)
├── tests/test_detectors.py # pytest tests
├── scripts/make_sample_video.py
├── notebooks/              # original Colab notebook
├── requirements.txt        # Python dependencies (CPU PyTorch)
├── packages.txt            # system packages for Streamlit Cloud
└── .streamlit/config.toml
```

## Run locally

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** → choose your repo, branch `main`, main file `app.py`.
4. In **Advanced settings** choose **Python 3.11 or 3.12**.
5. Click **Deploy**. The first build takes a few minutes (PyTorch install).

## Tests

```bash
pip install pytest
pytest -q tests
```

## Notes

- No dataset download is needed: the YOLO model is already trained on COCO.
- The app runs on CPU; use the nano model for speed.
- The camera feature uses `st.camera_input` (a snapshot, not a live stream) and
  needs HTTPS, which Streamlit Cloud provides.

## License

MIT
