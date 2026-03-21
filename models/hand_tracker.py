import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands = 1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        def detect(self, frame):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)

            if not res.multi_hands_landmarks:
                return None
            hand = res.multi_hands_landmarks[0]

            keypoints = []
            for landmark in hand.landmark:
                keypoints.append((landmark.x, landmark.y, landmark.z))

            return keypoints