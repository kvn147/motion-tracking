import cv2

# Hand connections from Medapipe
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

def draw_landmarks(frame, results):
    if not results.hand_landmarks:
        return frame

    h, w, _ = frame.shape

    for hand_landmarks in results.hand_landmarks:
        points = []

        for lm in hand_landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            points.append((x, y))
            cv2.circle(frame, (x, y), 3, (0,255,0), -1)

        # Draw connections
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (255,0,0), 2)

    return frame


def draw_debug(frame, servo_values=None, fps=None):
    y = 30

    if servo_values:
        for name, val in servo_values.items():
            cv2.putText(frame, f"{name}: {val}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            y += 25

    if fps:
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    return frame