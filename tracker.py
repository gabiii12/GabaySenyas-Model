import csv
import os
import sys

import cv2
import mediapipe as mp
import numpy as np

if len(sys.argv) != 3:
    print("Usage: python collect_landmarks.py <label> <signer_id>")
    sys.exit(1)

LABEL, SIGNER = sys.argv[1], sys.argv[2]
OUT = "data/landmarks.csv"
os.makedirs("data", exist_ok=True)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def normalize(hand_landmarks):
    pts = np.array([[l.x, l.y, l.z] for l in hand_landmarks.landmark])
    pts -= pts[0]  # wrist -> origin
    scale = np.abs(pts).max()
    if scale > 0:
        pts /= scale
    return pts.flatten()


cap = cv2.VideoCapture(0)
recording = False
count = 0
new_file = not os.path.exists(OUT)

with open(OUT, "a", newline="") as f, mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6
) as hands:
    writer = csv.writer(f)
    if new_file:
        writer.writerow(["signer", "label"] + [f"p{i}" for i in range(63)])

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror view
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            if recording:
                writer.writerow([SIGNER, LABEL] + list(normalize(hand)))
                count += 1

        status = "REC" if recording else "PAUSED"
        cv2.putText(frame, f"{LABEL} | {SIGNER} | {status} | {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if recording else (255, 255, 255), 2)
        cv2.imshow("Collect FSL landmarks", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            recording = not recording
        elif key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print(f"Saved {count} samples for '{LABEL}' ({SIGNER}) to {OUT}")