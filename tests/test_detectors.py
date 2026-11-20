import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.detectors import detect_moving_objects, read_video_frames  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def test_motion_detects_moving_box():
    frames = []
    for i in range(40):
        f = np.full((480, 640, 3), 60, np.uint8)
        if i >= 12:
            x = 30 + (i - 12) * 15
            f[190:330, x:x + 130] = (200, 120, 40)
        frames.append(f)
    results = detect_moving_objects(frames)
    assert len(results) == 40
    assert results[5][2] == 0                      # nothing moves yet
    assert any(r[2] > 0 for r in results[12:])     # box detected later


def test_sample_video_readable():
    frames = read_video_frames(str(ROOT / "samples" / "sample_motion.mp4"))
    assert len(frames) > 10
