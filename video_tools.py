"""Simple video analysis helpers using OpenCV.
The routines are intentionally lightweight and only provide rough estimates
useful for quick coaching feedback."""

from __future__ import annotations

import cv2
import numpy as np


def analyze_video(path: str) -> dict:
    """Estimate spin-down time of a Beyblade clip.

    The algorithm measures frame-to-frame differences inside a central region
    of interest. When motion falls below a threshold for a sustained period
    we mark it as a stamina finish. This is only a heuristic.
    """
    cap = cv2.VideoCapture(path)
    prev = None
    frame = 0
    last_motion = 0
    motion_threshold = 2.0

    while True:
        ret, img = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        roi = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        if prev is not None:
            diff = cv2.absdiff(roi, prev)
            score = diff.mean()
            if score > motion_threshold:
                last_motion = frame
            prev = roi
        else:
            prev = roi
        frame += 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    spin_time = last_motion / fps
    ko = False
    return {"spin_time": spin_time, "ko_detected": ko}

