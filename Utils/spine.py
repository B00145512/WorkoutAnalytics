from Utils.show_points_and_lines import connect_landmarks, draw_points


def draw_spine(image, results):
    if results.pose_landmarks:
        h, w, _ = image.shape
        lm = results.pose_landmarks.landmark

        left_shoulder = lm[11]
        right_shoulder = lm[12]
        left_hip = lm[23]
        right_hip = lm[24]

        pts = {'left_shoulder': left_shoulder,
               'right_shoulder': right_shoulder,
               'left_hip': left_hip,
               'right_hip': right_hip}

        neck = (int(((left_shoulder.x + right_shoulder.x)/2)*w),
                int(((left_shoulder.y + right_shoulder.y)/2)*h)
        )
        stomach = (int(((left_hip.x + right_hip.x)/2)*w),
                   int(((left_hip.y + right_hip.y)/2)*h)
        )


        connect_landmarks(image, left_shoulder, right_shoulder, w ,h)
        connect_landmarks(image, neck, stomach, w ,h)
        for name, l in pts.items():
            draw_points(image, l, name, w, h)