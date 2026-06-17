import cv2

def to_pixel(point, w, h):
    if hasattr(point, 'x'):
        return int(point.x * w), int(point.y * h)

    return point

def draw_points(image, l, label, w, h):
    x_px, y_px = int(l.x * w), int(l.y * h)
    cv2.circle(image, (x_px, y_px), 6, (0, 255, 0), -1)
    cv2.putText(image, label, (x_px + 6, y_px - 6),
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def connect_landmarks(image, point1, point2, w, h):
        x1, y1 = to_pixel(point1, w, h)
        x2, y2 = to_pixel(point2, w, h)

        cv2.line(image, (x1, y1), (x2, y2), (100, 255, 100), 2)