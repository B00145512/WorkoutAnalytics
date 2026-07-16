
import numpy as np
def find_velocity(cur_point, prev_point, dt):
    """
    Find velocity usin pythagoras
    cur_point: current point (x, y)
    prev_point: previous point (x, y)
    dt: time difference between current and previous point
    """
    if prev_point is None or dt <= 0:
        return 0
    
    dx = cur_point[0] - prev_point[0]
    dy = cur_point[1] - prev_point[1]
    distance = pyth(dx, dy)

    return distance/dt

def find_velocity_nodt(cur_point, prev_point):
    if prev_point is None:
        return 0
    
    dx = cur_point[0] - prev_point[0]
    dy = cur_point[1] - prev_point[1]
    distance = pyth(dx, dy)

    return distance

def pyth(dx, dy):
    """
    Find distance using pythagoras
    dx: difference in x coordinates
    dy: difference in y coordinates
    """
    return np.sqrt(dx ** 2 + dy ** 2)