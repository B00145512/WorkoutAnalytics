from Utils.FindDistance import find_distance
from Utils.show_points_and_lines import connect_landmarks, draw_points
import numpy as np

# Finds angle of torso lean based on shoulder and hip coordinates
def find_torso_angle(shoulder, hip):
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]
    angle = np.degrees(np.arctan2(abs(dx), abs(dy)))

    return angle

# Draws the spine on the image and checks if the back is straight based on the baseline spine distance
def draw_spine(image, results, baseline_spine_dist):
    MIN_RATIO = 0.85
    MAX_RATIO = 1.15
    MAX_TORSO_LEAN = 5

    if not results.pose_landmarks:
        return None

    h, w, _ = image.shape
    lm = results.pose_landmarks.landmark

    left_shoulder = lm[11]
    right_shoulder = lm[12]
    left_hip = lm[23]
    right_hip = lm[24]

    # Midpoints
    neck = (
        int(((left_shoulder.x + right_shoulder.x) / 2) * w),
        int(((left_shoulder.y + right_shoulder.y) / 2) * h)
    )
    stomach = (
        int(((left_hip.x + right_hip.x) / 2) * w),
        int(((left_hip.y + right_hip.y) / 2) * h)
    )

    current_spine_dist = find_distance(neck, stomach)

    if baseline_spine_dist is None:
        baseline_spine_dist = current_spine_dist

    if baseline_spine_dist > 0:
        ratio = current_spine_dist / baseline_spine_dist
    else:
        ratio = 1.0

    # torso lean
    torso_lean = find_torso_angle(neck, stomach)

    front_lean_check = MIN_RATIO <= ratio <= MAX_RATIO
    back_is_straight = front_lean_check and torso_lean <= MAX_TORSO_LEAN

    # colour based on whether the back is straight or not
    color = (100, 255, 100) if back_is_straight else (0, 0, 255)

    connect_landmarks(image, left_shoulder, right_shoulder, w, h, color=color)
    connect_landmarks(image, neck, stomach, w, h, color=color)

    for p in [left_shoulder, right_shoulder, left_hip, right_hip]:
        draw_points(image, p, "", w, h)

    return baseline_spine_dist, ratio, back_is_straight