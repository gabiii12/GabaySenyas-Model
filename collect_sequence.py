"""
Record sign CLIPS (sequences of frames) with both hands.

Usage:
    python collect_sequences.py <sign_name> <signer_id>
    e.g. python collect_sequences.py good_morning gab

Controls (click the camera window first):
    SPACE  start a clip, SPACE again to stop and save it
    C      cancel the clip you are recording right now (nothing is saved)
    U      undo: delete the last clip saved in this run (press again to go further back)
    Q      quit
"""
import glob
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

SEQ_LEN = 30
MIN_FRAMES = 10
MODEL_PATH = "model/hand_landmarker.task"

if len(sys.argv) != 3:
    print("Usage: python collect_sequences.py <sign_name> <signer_id>")
    print("Use underscores, no spaces. Example: python collect_sequences.py good_morning gab")
    sys.exit(1)
SIGN, SIGNER = sys.argv[1], sys.argv[2]

# Guard: the sign name must be a plain name, not a path (this caused nested folders before)
if any(c in SIGN for c in ("\\", "/", ":")):
    print(f"'{SIGN}' looks like a path. Give just the sign name, e.g. good_morning")
    sys.exit(1)

out_dir = os.path.join("data", "sequences", SIGN)
os.makedirs(out_dir, exist_ok=True)

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
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

existing = len(glob.glob(os.path.join(out_dir, f"{SIGNER}_*.npy")))
saved = 0
saved_paths = []  # clips saved in this run, newest last (for undo)
recording = False
frames, hand_seen = [], []
cap = cv2.VideoCapture(0)
t0 = time.time()

with vision.HandLandmarker.create_from_options(options) as detector:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ts = int((time.time() - t0) * 1000)
        result = detector.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        for lm in result.hand_landmarks:
            for a, b in CONNECTIONS:
                cv2.line(frame, (int(lm[a].x * w), int(lm[a].y * h)),
                         (int(lm[b].x * w), int(lm[b].y * h)), (0, 255, 0), 2)

        if recording:
            frames.append(frame_features(result))
            hand_seen.append(len(result.hand_landmarks) > 0)

        status = f"REC ({len(frames)} frames)" if recording else "READY"
        cv2.putText(frame, f"{SIGN} | {SIGNER} | {status} | saved: {saved}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255) if recording else (255, 255, 255), 2)
        cv2.putText(frame, "SPACE rec/save | C cancel | U undo last | Q quit",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.imshow("Record sign clips", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            if not recording:
                recording, frames, hand_seen = True, [], []
            else:
                recording = False
                if len(frames) >= MIN_FRAMES and np.mean(hand_seen) >= 0.6:
                    clip = resample(frames).astype("float32")
                    path = os.path.join(out_dir, f"{SIGNER}_{existing + saved}.npy")
                    np.save(path, clip)
                    saved_paths.append(path)
                    saved += 1
                else:
                    print("Clip discarded (too short or hands not visible enough).")
        elif key == ord("c") and recording:
            recording = False
            frames, hand_seen = [], []
            print("Recording cancelled.")
        elif key == ord("u") and not recording and saved_paths:
            os.remove(saved_paths.pop())
            saved -= 1
            print("Last clip deleted.")
        elif key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print(f"Saved {saved} clips for '{SIGN}' ({SIGNER}) in {out_dir}")