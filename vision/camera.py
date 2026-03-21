import cv2

class Camera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)

    def read(self):
        ret, frame = self.cap.read()

        if not ret:
            return None
        return frame
    
    def release(self):
        self.cap.release()

    