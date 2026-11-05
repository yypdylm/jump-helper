"""Clipped/pawn-occluded tops and measured pawn contact geometry."""

import math
from vision_pixels import (
    _color_contours,
    background_distance,
    bounded_pawn_mask,
    material_projection,
    row_background,
)
from vision_geometry import (
    _shape,
)
from vision_validation import (
    _exposed_upper_edge,
    _platform_candidates,
    _supported_small_surface,
    _trimmed_upper_edge,
)

def _clipped_diamond_candidates(image, origin, angle, allow_trim=False):
    """Infer one slightly clipped corner from three visible symmetric corners."""
    import cv2
    import numpy as np
    height, width = image.shape[:2]
    result = []
    for contour in _color_contours(image, 12, False):
        x, y, bw, bh = cv2.boundingRect(contour)
        frame_tolerance = max(1, round(width * 0.005)) if allow_trim else 0
        left_clip, right_clip = x <= frame_tolerance, x + bw >= width - frame_tolerance
        if left_clip == right_clip or y == 0 or y + bh >= height:
            continue
        if cv2.contourArea(contour) < width ** 2 * 0.008:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, perimeter * 0.02, True).reshape(-1, 2)
        # A shallow crop leaves a very short vertical cut. Coarse contour
        # simplification can erase one endpoint and pretend the top is a full
        # quadrilateral. Recover the cut with finer approximation, still
        # requiring exactly three visible corners and two boundary endpoints.
        for precision in (0.01, 0.005):
            if len(polygon) != 4:
                break
            polygon = cv2.approxPolyDP(contour, perimeter * precision, True).reshape(-1, 2)
        if len(polygon) != 5 or not cv2.isContourConvex(polygon):
            continue
        # Quantization/antialiasing can pull a cut endpoint inward by 2px,
        # or 3px when the material-rim fallback is independently verified.
        # With a verified rim, measure from the region's cut edge instead of
        # the screen edge; the gap to the screen is checked separately below.
        edge_tolerance = max(1, round(width * (0.0075 if allow_trim else 0.005)))
        cut_left = x if allow_trim else 0
        cut_right = x + bw - 1 if allow_trim else width - 1
        boundary = (polygon[:, 0] <= cut_left + edge_tolerance if left_clip else
                    polygon[:, 0] >= cut_right - edge_tolerance)
        if np.count_nonzero(boundary) != 2:
            continue
        if allow_trim and (x != 0 if left_clip else x + bw != width):
            # A one/two-pixel quantized rim may separate the surface from the
            # frame. It must be solid material, never a strip of visible sky.
            ends = polygon[boundary, 1]
            y1, y2 = int(ends.min()) + 2, int(ends.max()) - 1
            gap = image[y1:y2, :x] if left_clip else image[y1:y2, x + bw:]
            if gap.size == 0:
                continue
            region = np.zeros(image.shape[:2], np.uint8)
            cv2.fillPoly(region, [contour], 255)
            base = np.median(image[region > 0], axis=0).astype(float)
            norm = float(np.dot(base, base))
            if norm < 100:
                continue
            pixels = gap.reshape(-1, 3).astype(float)
            brightness, error = material_projection(pixels, base, norm)
            if np.mean((brightness > 0.85) & (brightness < 1.1) & (error < 8)) < 0.97:
                continue
        visible = polygon[~boundary]
        top, bottom = visible[visible[:, 1].argmin()], visible[visible[:, 1].argmax()]
        side = visible[visible[:, 0].argmax() if left_clip else visible[:, 0].argmin()]
        if np.array_equal(side, top) or np.array_equal(side, bottom):
            continue
        cx, cy = (top + bottom) / 2
        if abs(float(top[0] - bottom[0])) > width * 0.015 or abs(float(side[1] - cy)) > width * 0.01:
            continue
        missing = np.array([2 * cx - side[0], 2 * cy - side[1]])
        overshoot = -missing[0] if left_clip else missing[0] - (width - 1)
        # A corner exactly on the last pixel is also valid: rounding of
        # the three visible corners can put a shallow crop on that boundary.
        if not 0 <= overshoot <= width * 0.06:
            continue
        full = cv2.convexHull(np.rint([top, bottom, side, missing]).astype(np.int32))
        full_area = cv2.contourArea(full)
        fraction = cv2.contourArea(contour) / max(1, full_area)
        if not 0.9 <= fraction <= 1.03 or _shape(full, width, angle) != "diamond":
            continue
        if not _exposed_upper_edge(full, image):
            if not allow_trim or not _trimmed_upper_edge(full, image):
                continue
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, full, 255)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8)) > 0
        background = row_background(image)
        if np.mean(background_distance(image, background)[mask] < 12) >= 0.02:
            continue
        # Only this checked reconstruction may extend beyond the image. Its
        # landing point must still lie well inside the visible part of the top.
        detector = "clipped-diamond-trim" if allow_trim else "clipped-diamond"
        candidates = _platform_candidates([full], origin, width, angle, detector)
        for candidate in candidates:
            px, py = candidate["projection"]["point"]
            if not width * 0.02 < px < width * 0.98 or not 0 < py < height:
                continue
            candidate["visible_fraction"] = min(1., fraction)
            candidate["clipped_edge"] = "left" if left_clip else "right"
            result.append(candidate)
    return result

def _pawn_occluded_diamond_contours(image, origin, angle):
    """Recover a top split by the pawn from three visible symmetric corners.

    The fourth corner can be hidden or still visible beyond the pawn. In both
    cases the entire missing region must be explained by material or pawn,
    with an exposed outer boundary and a supporting body.
    """
    import cv2
    import numpy as np
    height, width = image.shape[:2]
    pawn = bounded_pawn_mask(image, origin)
    soft_pawn = cv2.dilate(pawn, np.ones((3, 3), np.uint8)) > 0
    background = row_background(image)
    background_like = background_distance(image, background) < 12
    result = []
    for contour in _color_contours(image, 12, False):
        area = cv2.contourArea(contour)
        if not width ** 2 * 0.004 < area < width ** 2 * 0.18:
            continue
        hull = cv2.convexHull(contour)
        polygon = cv2.approxPolyDP(hull, 0.025 * cv2.arcLength(hull, True), True).reshape(-1, 2)
        if not 5 <= len(polygon) <= 7:
            continue
        for index in range(len(polygon)):
            a, b, c = (polygon[(index + j) % len(polygon)].astype(float) for j in range(3))
            # Three consecutive real corners must define two full symmetric
            # edges. Validate the inferred corner against pixels below, too:
            # a pawn can disconnect the corner without covering it entirely.
            if (abs(a[0] - c[0]) > width * 0.005 or
                    abs(b[1] - (a[1] + c[1]) / 2) > width * 0.005):
                continue
            d = a + c - b
            dx, dy = map(round, d)
            if not (0 <= dx < width and 0 <= dy < height):
                continue
            quad = np.round([a, b, c, d]).astype(np.int32).reshape(-1, 1, 2)
            if _shape(quad, width, angle) != "diamond":
                continue
            region = np.zeros((height, width), np.uint8);surface = region.copy()
            cv2.fillPoly(region, [contour], 255);cv2.fillConvexPoly(surface, quad, 255)
            if np.count_nonzero((region > 0) & (surface > 0)) / max(1, np.count_nonzero(region)) < 0.95:
                continue
            interior = cv2.erode(surface, np.ones((3, 3), np.uint8)) > 0
            if np.mean(background_like[interior]) > 0.02:
                continue
            occupied = interior & (pawn > 0)
            if not 8 <= np.count_nonzero(occupied) <= np.count_nonzero(interior) * 0.12:
                continue
            missing = interior & (region == 0)
            if np.count_nonzero(missing) < 20:
                continue
            base = np.median(image[region > 0], axis=0)
            norm = np.dot(base, base)
            if norm < 100:
                continue
            pixels = image.astype(float)
            brightness, residual = material_projection(pixels, base, norm)
            if not pawn[dy, dx] and not (
                    0.85 < brightness[dy, dx] < 1.1 and residual[dy, dx] < 8):
                continue
            shade = (brightness > 0.45) & (brightness < 1.02) & (residual < 12)
            if np.mean((shade | soft_pawn)[missing]) < 0.97:
                continue
            # Top width varies independently of body height. For a wide top,
            # don't demand a pillar extending through the ground below it.
            top_width = cv2.boundingRect(quad)[2]
            support_depth = min(top_width * 0.5, width * 0.08)
            if not _exposed_upper_edge(quad, image, origin) or not _supported_small_surface(
                    quad, image, depth_ratio=support_depth / top_width):
                continue
            result.append(quad)
    return result

def _player_contact_origin(image, origin):
    """Locate the foot ellipse center from its visible lower arc, not its bottom pixel."""
    import cv2
    import numpy as np
    width = image.shape[1]
    ox, oy = map(round, origin)
    radius = round(width * 0.06)
    if ox-radius < 0 or ox+radius >= width or oy >= image.shape[0]:
        return None
    mask = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV),
                       (110, 50, 30), (170, 255, 160))
    rows = []
    for y in range(max(0, oy-round(width*0.05)), oy+1):
        xs = np.flatnonzero(mask[y, ox-radius:ox+radius+1]) + ox-radius
        if len(xs):
            if xs[-1]-xs[0]+1 != len(xs):
                # The last antialiased row can cross the purple hue cutoff
                # while blending into a colored supporting platform. Only
                # bridge bounded gaps explained by the pixels above/below.
                gaps = np.arange(xs[0], xs[-1]+1)[mask[y, xs[0]:xs[-1]+1] == 0]
                if (y != oy or y == 0 or y+1 >= image.shape[0] or
                        len(gaps) > max(1, round(width*0.01)) or
                        not np.all(mask[y-1, xs[0]:xs[-1]+1])):
                    return None
                upper = image[y-1, gaps].astype(float)
                lower = image[y+1, gaps].astype(float)
                pixel = image[y, gaps].astype(float)
                delta = upper-lower
                norm = np.sum(delta*delta, axis=1)
                blend = np.sum((pixel-lower)*delta, axis=1)/np.maximum(norm, 1)
                residual = np.max(abs(pixel-(lower+blend[:, None]*delta)), axis=1)
                if not np.all((norm > 400) & (blend > 0.25) & (blend < 0.95) & (residual < 8)):
                    return None
            rows.append((y, int(xs[0]), int(xs[-1])))
    if not rows:
        return None
    widest = max(b-a+1 for y,a,b in rows)
    start = max(y for y,a,b in rows if b-a+1 == widest)
    arc = [(y,a,b) for y,a,b in rows if y >= start]
    if not (width*0.035 <= widest <= width*0.09 and len(arc) >= 5 and
            arc[-1][0] == oy and arc[-1][2]-arc[-1][1]+1 < widest*0.6):
        return None
    if any(b-a > pb-pa+1 for (_,pa,pb),(_,a,b) in zip(arc,arc[1:])):
        return None
    points = np.array([(x,y) for y,a,b in arc for x in (a,b)],np.float32)
    (cx,cy),(a,b),degrees = cv2.fitEllipse(points.reshape(-1,1,2))
    minor,major = min(a,b),max(a,b)
    orientation = degrees if a >= b else degrees+90
    if not (major > 0 and 0.4 < minor/major < 0.75 and
            abs((orientation+90)%180-90) < 8 and
            abs(cx-origin[0]) <= width*0.005 and
            start-widest*0.12 <= cy <= start+1 and
            abs(cy+minor/2-oy) <= 1.5 and 0.9*widest < major < 1.1*widest):
        return None
    theta = math.radians(degrees)
    delta = points-(cx,cy)
    rx = delta[:,0]*math.cos(theta)+delta[:,1]*math.sin(theta)
    ry = -delta[:,0]*math.sin(theta)+delta[:,1]*math.cos(theta)
    error = abs(np.sqrt((rx/(a/2))**2+(ry/(b/2))**2)-1)
    if np.mean(error) > 0.06 or np.max(error) > 0.15:
        return None
    return (float(cx),float(cy))
