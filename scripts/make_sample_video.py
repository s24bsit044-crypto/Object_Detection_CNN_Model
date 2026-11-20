"""Create samples/sample_motion.mp4 (a box moving over a static background)."""
from pathlib import Path

import cv2
import numpy as np

out = Path(__file__).resolve().parent.parent / "samples" / "sample_motion.mp4"
writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 15, (640, 480))
for i in range(90):
    f = np.full((480, 640, 3), 60, np.uint8)
    if i >= 20:
        x = 30 + (i - 20) * 7
        cv2.rectangle(f, (x, 190), (x + 130, 330), (200, 120, 40), -1)
    writer.write(f)
writer.release()
print("Saved", out)
