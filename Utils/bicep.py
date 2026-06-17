##import Utils.find_angle as find_angle
##def draw_landmarks(image, results):
##    if results.pose_landmarks:
##        h, w, _ = image.shape
##        lm = results.pose_landmarks.landmark
##
##        left_shoulder = lm[11]
##        right_shoulder = lm[12]
##        left_elbow = lm[13]
##        right_elbow = lm[14]
##        left_wrist = lm[15]
##        right_wrist = lm[16]
##        #print("Shoulder: ",left_shoulder)
##        #print("Wrist: ",left_wrist)
##        #print("Elbow: ",left_elbow)
##        pts = {'L_shoulder': left_shoulder,
##                'R_shoulder': right_shoulder,
##                'L_elbow': left_elbow,
##                'R_elbow': right_elbow,
##                'L_wrist': left_wrist,
##                'R_wrist': right_wrist}
##        print(find_angle.find_angle([left_shoulder.x, left_shoulder.y], [left_elbow.x, left_elbow.y], [left_wrist.x, left_wrist.y]))
##        for name, l in pts.items():
##            draw_landmarks(image, l)