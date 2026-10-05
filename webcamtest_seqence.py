"""
REAL-TIME sign recognition (sequence model). No key presses needed.

Usage:
    python webcam_realtime.py

Press Q to quit.

How it works:
  - Keeps the last WINDOW_SEC seconds of hand data.
  - Every CHECK_EVERY seconds it resamples that window to 30 frames and asks the model.
  - A word is shown only if the SAME sign wins several checks in a row,
    with confidence >= THRESHOLD, and it is not the "none" class.

Tune these if needed:
  WINDOW_SEC  longer if your signs take longer, shorter for quick signs
  THRESHOLD   higher = fewer false detections, but misses more real ones
"""
import json
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

SEQ_LEN = 30
WINDOW_SEC = 1.5
CHECK_EVERY = 0.2
THRESHOLD = 0.85
VOTES_NEEDED = 3
SHOW_SEC = 1.5
NONE_LABEL = "none"

model = tf.keras.models.load_model("model/fsl_sequence_model.keras")
with open("model/sequence_labels.json") as f:
    labels = json.load(f)
if NONE_LABEL not in labels:
    print(f"WARNING: no '{NONE_LABEL}' class trained. It will name a sign for ANY movement.")

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def hand_features(lm):
    pts = np.array([[l.x, l.y, l.z] for l in lm])
    wrist_xy = pts[0, :2].copy()
    pts -= pts[0]
    scale = np.abs(pts).max()
    if scale > 0:
        pts /= scale
    return np.concatenate([pts.flatten(), wrist_xy])


def frame_features(result):
    feats = np.zeros(130, dtype="float32")
    for lm, handed in zip(result.hand_landmarks, result.handedness):
        slot = 0 if handed[0].category_name == "Left" else 1
        feats[slot * 65:(slot + 1) * 65] = hand_features(lm)
    return feats


def resample(frames, n=SEQ_LEN):
    arr = np.array(frames)
    src = np.arange(len(arr))
    dst = np.linspace(0, len(arr) - 1, n)
    return np.stack([np.interp(dst, src, arr[:, i]) for i in range(arr.shape[1])], axis=1)


options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="model/hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

buffer = deque()  # (time, features, hand_seen)
votes = deque(maxlen=VOTES_NEEDED)
shown_word, shown_until = "", 0.0
guess = ""
last_check = 0.0
cap = cv2.VideoCapture(0)
t0 = time.time()

with vision.HandLandmarker.create_from_options(options) as detector:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirrored, same as training
        h, w = frame.shape[:2]
        now = time.time() - t0
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = detector.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(now * 1000))

        for lm in result.hand_landmarks:
            for a, b in CONNECTIONS:
                cv2.line(frame, (int(lm[a].x * w), int(lm[a].y * h)),
                         (int(lm[b].x * w), int(lm[b].y * h)), (0, 255, 0), 2)

        buffer.append((now, frame_features(result), len(result.hand_landmarks) > 0))
        while buffer and now - buffer[0][0] > WINDOW_SEC:
            buffer.popleft()

        if now - last_check >= CHECK_EVERY:
            last_check = now
            seen = [s for _, _, s in buffer]
            idx = [k for k, s in enumerate(seen) if s]
            if len(buffer) >= 10 and idx and np.mean(seen) >= 0.5:
                feats = [f for _, f, _ in buffer][idx[0]: idx[-1] + 1]  # trim empty ends
                if len(feats) >= 10:
                    clip = resample(feats).astype("float32")[None]
                    probs = model.predict(clip, verbose=0)[0]
                    best = int(np.argmax(probs))
                    guess = f"{labels[best]} {probs[best] * 100:.0f}%"
                    if probs[best] >= THRESHOLD and labels[best] != NONE_LABEL:
                        votes.append(labels[best])
                    else:
                        votes.clear()
                    if len(votes) == VOTES_NEEDED and len(set(votes)) == 1:
                        shown_word, shown_until = votes[0], now + SHOW_SEC
                        votes.clear()
                        buffer.clear()  # start fresh so one sign isn't counted twice
                else:
                    votes.clear()
            else:
                votes.clear()
                guess = ""

        if now < shown_until:
            cv2.putText(frame, shown_word, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.8, (0, 0, 255), 4)
        cv2.putText(frame, f"guess: {guess}", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 0), 2)
        cv2.imshow("Real-time FSL", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()