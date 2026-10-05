"""
Live webcam test for the trained sign model.

Usage:
    python webcam_test.py

Needs (created earlier):
    model/fsl_model.keras
    model/labels.json
    model/hand_landmarker.task

Press Q to quit.
"""
import json
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

model = tf.keras.models.load_model("model/fsl_model.keras")
with open("model/labels.json") as f:
    labels = json.load(f)

options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="model/hand_landmarker.task"),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.6,
)

# Standard 21-point hand skeleton, for drawing
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def normalize(landmarks):
    pts = np.array([[l.x, l.y, l.z] for l in landmarks])
    pts -= pts[0]
    scale = np.abs(pts).max()
    if scale > 0:
        pts /= scale
    return pts.flatten()


recent = deque(maxlen=7)  # smooth flickering predictions
cap = cv2.VideoCapture(0)

with vision.HandLandmarker.create_from_options(options) as detector:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirrored, same as training
        h, w = frame.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

        text = "No hand"
        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            for a, b in CONNECTIONS:
                cv2.line(frame, (int(lm[a].x * w), int(lm[a].y * h)),
                         (int(lm[b].x * w), int(lm[b].y * h)), (0, 255, 0), 2)
            probs = model.predict(normalize(lm).reshape(1, -1).astype("float32"), verbose=0)[0]
            idx = int(np.argmax(probs))
            recent.append(labels[idx])
            letter = Counter(recent).most_common(1)[0][0]
            text = f"{letter}  ({probs[idx] * 100:.0f}%)"
        else:
            recent.clear()

        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imshow("FSL test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()