import cv2

def draw_landmark(image, l, label, w, h):
    x_px, y_px = int(l.x * w), int(l.y * h)
    cv2.circle(image, (x_px, y_px), 6, (0, 255, 0), -1)
    cv2.putText(image, label, (x_px + 6, y_px - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)