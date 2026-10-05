"""
Convert existing VIDEO clips of one sign into landmark sequences.

Usage:
    python videos_to_sequences.py <video_folder> <sign_name> [signer_id] [--noflip]
    e.g. python videos_to_sequences.py "E:/clips/good_morning" good_morning gab

Each video file = ONE performance of the sign (one clip = one sample).
Output: data/sequences/<sign_name>/<signer_id>_<n>.npy   shape (30, 130)
Same format as collect_sequences.py, so train_sequences.py reads both.

Notes:
  - Frames are mirrored by default to match the mirrored webcam view used
    everywhere else. If your videos were already saved mirrored (selfie view),
    add --noflip.
  - Leading/trailing frames with no hands are trimmed.
  - Use a different signer_id for each different person (train_sequences.py
    uses it to test on people the model hasn't seen).
"""
import glob
import os
import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

SEQ_LEN = 30
MIN_FRAMES = 10
MODEL_PATH = "model/hand_landmarker.task"
EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")

args = [a for a in sys.argv[1:] if not a.startswith("--")]
FLIP = "--noflip" not in sys.argv
if len(args) < 2:
    print("Usage: python videos_to_sequences.py <video_folder> <sign_name> [signer_id] [--noflip]")
    sys.exit(1)
FOLDER, SIGN = args[0], args[1]
SIGNER = args[2] if len(args) > 2 else "video"

out_dir = os.path.join("data", "sequences", SIGN)
os.makedirs(out_dir, exist_ok=True)


def hand_features(lm):
    pts = np.array([[l.x, l.y, l.z] for l in lm])
    wrist_xy = pts[0, :2].copy()
    pts -= pts[0]
    scale = np.abs(pts).max()
    if scale > 0:
        pts /= scale
    return np.concatenate([pts.flatten(), wrist_xy])  # 65 values


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


def make_options():
    return vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


files = sorted(f for f in glob.glob(os.path.join(FOLDER, "*")) if f.lower().endswith(EXTS))
if not files:
    raise SystemExit(f"No video files found in {FOLDER}")

n_existing = len(glob.glob(os.path.join(out_dir, f"{SIGNER}_*.npy")))
saved = skipped = 0

for path in files:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames, seen = [], []
    i = 0
    # new detector per video so timestamps restart cleanly
    with vision.HandLandmarker.create_from_options(make_options()) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if FLIP:
                frame = cv2.flip(frame, 1)
            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(i * 1000 / fps))
            frames.append(frame_features(res))
            seen.append(len(res.hand_landmarks) > 0)
            i += 1
    cap.release()

    # trim frames with no hands at the start and end
    idx = [k for k, s in enumerate(seen) if s]
    if idx:
        frames = frames[idx[0]: idx[-1] + 1]
        seen = seen[idx[0]: idx[-1] + 1]

    if len(frames) < MIN_FRAMES or np.mean(seen) < 0.6:
        print(f"SKIPPED {os.path.basename(path)} (too short or hands not visible enough)")
        skipped += 1
        continue

    np.save(os.path.join(out_dir, f"{SIGNER}_{n_existing + saved}.npy"),
            resample(frames).astype("float32"))
    saved += 1
    print(f"OK      {os.path.basename(path)} ({len(frames)} frames)")

print(f"\nDone: {saved} saved, {skipped} skipped -> {out_dir}")