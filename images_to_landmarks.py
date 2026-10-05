"""
Convert an image dataset into hand-landmark rows (MediaPipe Tasks API).

Expected folder structure (folder name = label):
    dataset/
        A/ img1.jpg img2.jpg ...
        B/ ...

Usage:
    python images_to_landmarks.py "E:/path/to/dataset"

Output: data/landmarks_dataset.csv

Each image is processed twice (original + mirrored) so the model works for
both hand orientations and matches the mirrored webcam view.

The hand model file is downloaded automatically on first run to
model/hand_landmarker.task
"""
import csv
import os
import sys
import urllib.request
from collections import Counter

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = "model/hand_landmarker.task"
OUT = "data/landmarks_dataset.csv"
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

if len(sys.argv) != 2:
    print("Usage: python images_to_landmarks.py <dataset_folder>")
    sys.exit(1)
ROOT = sys.argv[1]

os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("Downloading hand model (one time)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")


def normalize(landmarks):
    pts = np.array([[l.x, l.y, l.z] for l in landmarks])
    pts -= pts[0]  # wrist -> origin
    scale = np.abs(pts).max()
    if scale > 0:
        pts /= scale
    return pts.flatten()


options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
)

saved = Counter()
failed = Counter()

with open(OUT, "w", newline="") as f, vision.HandLandmarker.create_from_options(options) as detector:
    writer = csv.writer(f)
    writer.writerow(["signer", "label"] + [f"p{i}" for i in range(63)])

    labels = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
    for label in labels:
        folder = os.path.join(ROOT, label)
        files = [x for x in os.listdir(folder) if x.lower().endswith(EXTS)]
        print(f"{label}: processing {len(files)} images...", flush=True)
        for n, name in enumerate(files, 1):
            img = cv2.imread(os.path.join(folder, name))
            if img is None:
                failed[label] += 1
                continue
            found = False
            for variant in (img, cv2.flip(img, 1)):
                rgb = np.ascontiguousarray(cv2.cvtColor(variant, cv2.COLOR_BGR2RGB))
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = detector.detect(mp_img)
                if res.hand_landmarks:
                    writer.writerow(["dataset", label] + list(normalize(res.hand_landmarks[0])))
                    saved[label] += 1
                    found = True
            if not found:
                failed[label] += 1
            if n % 500 == 0:
                print(f"   {n}/{len(files)}", flush=True)
        print(f"{label}: {saved[label]} samples saved, {failed[label]} images with no hand found")

print(f"\nDone. Total samples: {sum(saved.values())} -> {OUT}")