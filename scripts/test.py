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
from control.kinematics import calculate_finger_angles
from control.filtering import MovingAverage

cam = Camera()
tracker = HandTracker()

# Initialize moving average filters for each finger
angle_filters = {
    "thumb": MovingAverage(window=5),
    "index": MovingAverage(window=5),
    "middle": MovingAverage(window=5),
    "ring": MovingAverage(window=5),
    "pinky": MovingAverage(window=5)
}

prev_time = time.time()

while True:
    frame = cam.read()
    if frame is None:
        break

    frame = cv2.flip(frame, 1)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    results = tracker.hands.detect(mp_image)

    finger_angles = calculate_finger_angles(results.hand_landmarks[0]) if results.hand_landmarks else None
    
    # Apply moving average filter to smooth angles
    if finger_angles:
        filtered_angles = {}
        for finger_name, angle in finger_angles.items():
            if finger_name in angle_filters:
                filtered_angles[finger_name] = angle_filters[finger_name].update(angle)
            else:
                filtered_angles[finger_name] = angle
        finger_angles = filtered_angles

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Draw
    frame = draw_landmarks(frame, results)
    frame = draw_debug(frame, fps=fps, finger_angles=finger_angles)

    cv2.imshow("Hand Tracking Debug", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
