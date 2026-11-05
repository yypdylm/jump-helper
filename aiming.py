"""Fixed-angle aiming geometry; image dependencies are loaded only in the GUI worker."""
import math


def project_target(origin, cursor, angle_degrees):
    """Orthogonal projection onto the up-left/up-right ray chosen by cursor side.

    Image y grows downward. Return horizontal span in the same units as the
    input, so the existing coefficient convention remains C * horizontal / W.
    """
    if not math.isfinite(angle_degrees) or not 5 <= angle_degrees <= 75:
        raise ValueError("角度须为相对水平线向上的 5–75 度。")
    dx, dy = cursor[0] - origin[0], cursor[1] - origin[1]
    if dx == 0:
        raise ValueError("请在棋子的左上方或右上方选择目标。")
    theta = math.radians(angle_degrees)
    direction = 1 if dx > 0 else -1
    ux, uy = direction * math.cos(theta), -math.sin(theta)
    along = dx * ux + dy * uy
    if along <= 0:
        raise ValueError("目标须在向左上或右上延伸的方向线上。")
    point = (origin[0] + along * ux, origin[1] + along * uy)
    return {"point": point, "horizontal": abs(along * ux), "along": along,
            "offset": math.hypot(cursor[0] - point[0], cursor[1] - point[1]),
            "direction": direction, "angle_degrees": angle_degrees}


def find_player(image):
    """Conservative purple head/body pairing. Ambiguity falls back to manual input."""
    import cv2
    import numpy as np
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (110, 50, 30), (170, 255, 160))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    candidates = []
    for i in range(1, count):
        x, y, w, h, area = map(int, stats[i])
        if not (0.035 * width <= w <= 0.12 * width and
                1.3 <= h / w <= 3.5 and area >= 0.002 * width * width and
                0.18 * height < y < 0.9 * height):
            continue
        for j in range(1, count):
            if j == i:
                continue
            hx, hy, hw, hh, ha = map(int, stats[j])
            if (0.025 * width <= hw <= 0.1 * width and 0.65 <= hh / hw <= 1.4 and
                    ha >= 0.0007 * width * width and
                    abs(centers[j][0] - centers[i][0]) <= 0.025 * width and
                    0 <= y - (hy + hh) <= 0.045 * width):
                ys, xs = np.where(labels == i)
                bottom = ys.max()
                bottom_x = float(np.median(xs[ys >= bottom - max(1, h // 8)]))
                candidates.append((bottom_x, float(bottom)))
                break
    return candidates[0] if len(candidates) == 1 else None


def game_window(items, window_id=None):
    candidates = [w for w in items if w.layer == 0 and w.width > 100 and w.height > 100 and
                  (w.id == window_id if window_id is not None else "跳一跳" in w.title)]
    if len(candidates) != 1:
        raise RuntimeError("未找到唯一的跳一跳窗口。请打开并露出游戏；有多个窗口时用 --window-id 指定。")
    return candidates[0]


def draw_guides(image, origin, hovered, angle_degrees):
    import cv2
    result = image.copy()
    if origin is None:
        cv2.putText(result, "Click player base", (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 90, 230), 2)
        return result, None
    ox, oy = origin
    span = image.shape[0] + image.shape[1]
    for sign in (-1, 1):
        endpoint = (round(ox + sign * span * math.cos(math.radians(angle_degrees))),
                    round(oy - span * math.sin(math.radians(angle_degrees))))
        cv2.line(result, (round(ox), round(oy)), endpoint, (190, 160, 30), 1)
    cv2.circle(result, (round(ox), round(oy)), 5, (0, 0, 255), 2)
    projection = None
    if hovered is not None:
        try:
            projection = project_target(origin, hovered, angle_degrees)
            end = tuple(round(v) for v in projection["point"])
            cv2.line(result, (round(ox), round(oy)), end, (0, 200, 0), 2)
            cv2.circle(result, end, 6, (0, 200, 0), 2)
            cv2.line(result, tuple(round(v) for v in hovered), end, (0, 140, 255), 1)
        except ValueError:
            pass
    cv2.putText(result, f"{angle_degrees:g} deg | click target | R: player", (8, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 200), 1)
    return result, projection
