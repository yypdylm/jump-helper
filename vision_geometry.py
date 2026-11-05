"""Shared platform geometry; observed boundaries remain detector-specific."""
import math


def edge_angles(polygon):
    import numpy as np
    return [math.degrees(math.atan2(abs(float(b[1] - a[1])), abs(float(b[0] - a[0]))))
            for a, b in zip(polygon, np.roll(polygon, -1, axis=0))]


def slanted_line_groups(lines, angle, tolerance):
    """Intercepts grouped by direction; preserve Hough observation order."""
    slope = math.tan(math.radians(angle))
    groups = {1: [], -1: []}
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = float(x2 - x1), float(y2 - y1)
        if dx == 0 or abs(math.degrees(math.atan2(abs(dy), abs(dx))) - angle) > tolerance:
            continue
        sign = 1 if dy / dx > 0 else -1
        intercept = ((y1 - sign * slope * x1) + (y2 - sign * slope * x2)) / 2
        groups[sign].append(intercept)
    return groups


def diamond_vertices(low_a, high_a, low_b, high_b, slope):
    """Intersect two pairs of parallel edges without rounding their vertices."""
    import numpy as np
    return np.array([[(low_b - low_a) / (2 * slope), (low_b + low_a) / 2],
                     [(high_b - low_a) / (2 * slope), (high_b + low_a) / 2],
                     [(high_b - high_a) / (2 * slope), (high_b + high_a) / 2],
                     [(low_b - high_a) / (2 * slope), (low_b + high_a) / 2]])


def edge_samples(a, b, start, stop):
    import numpy as np
    return a + np.linspace(start, stop, 40)[:, None] * (b - a)


def narrow_seam(rectangle, pixel_count, surface_width, angle):
    """The same coherent strip geometry is required for ink and material seams."""
    _, (a, b), degrees = rectangle
    major, minor = max(a, b), min(a, b)
    if minor < 2 or major < surface_width * 0.5 or minor / major > 0.25:
        return False
    if pixel_count / (major * minor) < 0.7:
        return False
    direction = degrees if a >= b else degrees + 90
    direction = abs((direction + 90) % 180 - 90)
    return abs(direction - angle) <= 6



def _shape(contour, width, angle):
    import cv2
    import numpy as np
    perimeter = cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True).reshape(-1, 2)
    x, y, bw, bh = cv2.boundingRect(contour)
    if len(polygon) == 4 and cv2.isContourConvex(polygon):
        # A platform top has four slanted edges; boxes/dialogs and side faces do not.
        slopes = edge_angles(polygon)
        if all(abs(s - angle) <= 13 for s in slopes):
            left, right = polygon[polygon[:, 0].argmin()], polygon[polygon[:, 0].argmax()]
            top, bottom = polygon[polygon[:, 1].argmin()], polygon[polygon[:, 1].argmax()]
            if abs(float(left[1] - right[1])) < 0.17 * bh and abs(float(top[0] - bottom[0])) < 0.17 * bw:
                return "diamond"
    if len(polygon) >= 6 and len(contour) >= 12:
        ellipse = cv2.fitEllipse(contour)
        (cx, cy), (a, b), degrees = ellipse
        if min(a, b) <= 0:
            return None
        # Reject tilted/skinny oval buttons and shapes which merely have rounded corners.
        if not 0.4 <= min(a, b) / max(a, b) <= 0.7 or not 0.4 <= bh / bw <= 0.72 or not 0.65 <= cv2.contourArea(contour) / (bw * bh) <= 0.9:
            return None
        theta = math.radians(degrees)
        points = contour.reshape(-1, 2).astype(float) - (cx, cy)
        rx = points[:, 0] * math.cos(theta) + points[:, 1] * math.sin(theta)
        ry = -points[:, 0] * math.sin(theta) + points[:, 1] * math.cos(theta)
        radius = np.sqrt((rx / (a / 2)) ** 2 + (ry / (b / 2)) ** 2)
        major_angle = degrees if a >= b else degrees + 90
        horizontal_error = abs((major_angle + 90) % 180 - 90)
        # Mean error alone can accept a cylinder wall merged with its shadow:
        # most points look oval, while a few long side segments are far off.
        radial_error = abs(radius - 1)
        tail_limit = 0.12 if bw <= width * 0.12 else 0.10
        if (horizontal_error < 12 and float(np.mean(radial_error)) < 0.05 and
                float(np.percentile(radial_error, 95)) < tail_limit):
            return "ellipse"
    return None
