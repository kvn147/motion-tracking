from pathlib import Path
import sys

import cv2
import time

import mediapipe as mp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision.camera import Camera
from models.hand_tracker import HandTracker
from vision.visualization import draw_landmarks, draw_debug

cam = Camera()
tracker = HandTracker()

prev_time = time.time()

while True:
    frame = cam.read()
    if frame is None:
        break

    frame = cv2.flip(frame, 1)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    results = tracker.hands.detect(mp_image)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Draw
    frame = draw_landmarks(frame, results)
    frame = draw_debug(frame, fps=fps)

    cv2.imshow("Hand Tracking Debug", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
