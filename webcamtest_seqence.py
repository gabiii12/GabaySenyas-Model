"""
Combined real-time test: static letters + motion signs.

Usage:
    python webcam_combined.py

Press Q to quit.

Needs:
    model/fsl_model.keras            + model/labels.json            (static alphabet)
    model/fsl_sequence_model.keras   + model/sequence_labels.json   (motion signs)
    model/hand_landmarker.task

Rule: if the wrists barely move in the last WINDOW_SEC seconds -> static model.
      if they move more than MOTION_THRESH            -> sequence model.
"""
import json
import os
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

SEQ_LEN = 30
WINDOW_SEC = 3.0        # how much history the sequence model sees
CHECK_EVERY = 0.2
MOTION_THRESH = 0.05    # wrist movement (fraction of image) that counts as "moving"
SEQ_THRESHOLD = 0.85
SEQ_VOTES = 3
STATIC_THRESHOLD = 0.90
STATIC_VOTES = 5        # ~1 second of a steady hand shape
SHOW_SEC = 1.5
NONE_LABEL = "none"

for p in ("model/fsl_model.keras", "model/labels.json",
          "model/fsl_sequence_model.keras", "model/sequence_labels.json",
          "model/hand_landmarker.task"):
    if not os.path.exists(p):
        raise SystemExit(f"Missing {p}. Train that model first.")

static_model = tf.keras.models.load_model("model/fsl_model.keras")
static_labels = json.load(open("model/labels.json"))
seq_model = tf.keras.models.load_model("model/fsl_sequence_model.keras")
seq_labels = json.load(open("model/sequence_labels.json"))

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def normalize63(lm):
    pts = np.array([[l.x, l.y, l.z] for l in lm])
    pts -= pts[0]
    scale = np.abs(pts).max()
    if scale > 0:
        pts /= scale
    return pts.flatten()


def hand_features(lm):
    wrist_xy = np.array([lm[0].x, lm[0].y])
    return np.concatenate([normalize63(lm), wrist_xy])  # 65 values


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


def motion_amount(feats_list):
    """How much the wrists moved (fraction of image) over the window."""
    arr = np.array(feats_list)
    best = 0.0
    for slot in (0, 1):
        xy = arr[:, slot * 65 + 63: slot * 65 + 65]
        present = np.abs(xy).sum(axis=1) > 0
        if present.sum() >= 5:
            best = max(best, float(xy[present].std(axis=0).sum()))
    return best


options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="model/hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

buffer = deque()  # (time, features, hand_seen)
seq_votes = deque(maxlen=SEQ_VOTES)
static_votes = deque(maxlen=STATIC_VOTES)
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
        frame = cv2.flip(frame, 1)
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

            if len(buffer) < 10 or not idx or np.mean(seen) < 0.5:
                seq_votes.clear()
                static_votes.clear()
                guess = ""
            else:
                feats = [f for _, f, _ in buffer][idx[0]: idx[-1] + 1]
                moving = motion_amount(feats) >= MOTION_THRESH

                if moving and len(feats) >= 10:
                    static_votes.clear()
                    probs = seq_model.predict(
                        resample(feats).astype("float32")[None], verbose=0)[0]
                    best = int(np.argmax(probs))
                    guess = f"MOVING: {seq_labels[best]} {probs[best] * 100:.0f}%"
                    if probs[best] >= SEQ_THRESHOLD and seq_labels[best] != NONE_LABEL:
                        seq_votes.append(seq_labels[best])
                    else:
                        seq_votes.clear()
                    if len(seq_votes) == SEQ_VOTES and len(set(seq_votes)) == 1:
                        shown_word, shown_until = seq_votes[0], now + SHOW_SEC
                        seq_votes.clear()
                        buffer.clear()
                elif not moving and len(result.hand_landmarks) == 1:
                    seq_votes.clear()
                    x = normalize63(result.hand_landmarks[0]).reshape(1, -1).astype("float32")
                    probs = static_model.predict(x, verbose=0)[0]
                    best = int(np.argmax(probs))
                    guess = f"STILL: {static_labels[best]} {probs[best] * 100:.0f}%"
                    if probs[best] >= STATIC_THRESHOLD:
                        static_votes.append(static_labels[best])
                    else:
                        static_votes.clear()
                    if len(static_votes) == STATIC_VOTES and len(set(static_votes)) == 1:
                        shown_word, shown_until = static_votes[0], now + SHOW_SEC
                        static_votes.clear()
                else:
                    seq_votes.clear()
                    static_votes.clear()

        if now < shown_until:
            cv2.putText(frame, shown_word, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.8, (0, 0, 255), 4)
        cv2.putText(frame, guess, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 0), 2)
        cv2.imshow("FSL combined", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()