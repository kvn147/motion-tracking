from pathlib import Path
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandTracker:
    def __init__(self):
        model_path = Path(__file__).parent / 'hand_landmarker.task'
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(base_options=base_options)
        self.hands = vision.HandLandmarker.create_from_options(options)
        
    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        if not res.multi_hand_landmarks:
            return None
        hand = res.multi_hand_landmarks[0]

        keypoints = []
        for landmark in hand.landmark:
            keypoints.append((landmark.x, landmark.y, landmark.z))

        return keypoints
