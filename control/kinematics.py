finger_map = {
    "wrist": [0],
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20]
}

def extract_fingers(keypoints):
    """
    Extract finger data from 21 keypoints.
    
    Args:
        keypoints: List of 21 tuples (x, y, z) from hand landmark detection
        
    Returns:
        Dict with finger names as keys and lists of landmarks as values
    """
    if len(keypoints) != 21:
        raise ValueError(f"Expected 21 keypoints, got {len(keypoints)}")
    
    fingers = {}
    for finger_name, indices in finger_map.items():
        fingers[finger_name] = [keypoints[i] for i in indices]
    
    return fingers


def get_finger_tips(keypoints):
    """
    Get the tip coordinates (last point) of each finger.
    
    Args:
        keypoints: List of 21 tuples (x, y, z) from hand landmark detection
        
    Returns:
        Dict with finger names as keys and tip coordinates (x, y, z) as values
    """
    fingers = extract_fingers(keypoints)
    tips = {}
    for finger_name, points in fingers.items():
        if finger_name != "wrist" and points:
            tips[finger_name] = points[-1]  # Last point is the tip
    return tips


def calculate_finger_angles(keypoints):
    """
    Calculate bending angles for each finger based on consecutive joint positions.
    
    Args:
        keypoints: List of 21 NormalizedLandmark objects or tuples (x, y, z) from hand landmark detection
        
    Returns:
        Dict with finger names as keys and angle values as values
    """
    import math
    
    # Convert NormalizedLandmark objects to tuples if needed
    normalized_keypoints = []
    for kp in keypoints:
        if hasattr(kp, 'x') and hasattr(kp, 'y'):
            normalized_keypoints.append((kp.x, kp.y, kp.z if hasattr(kp, 'z') else 0))
        else:
            normalized_keypoints.append(kp)
    
    fingers = extract_fingers(normalized_keypoints)
    angles = {}
    
    for finger_name, points in fingers.items():
        if finger_name == "wrist" or len(points) < 3:
            continue
        
        # Calculate angle between first three joints
        p1, p2, p3 = points[0], points[1], points[2]
        
        # Vectors from p2 to p1 and p2 to p3
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        # Dot product and magnitudes
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 > 0 and mag2 > 0:
            cos_angle = dot / (mag1 * mag2)
            angle = math.acos(max(-1, min(1, cos_angle)))
            angles[finger_name] = math.degrees(angle)
        else:
            angles[finger_name] = 0
    
    return angles

