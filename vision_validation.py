"""Shared physical-boundary, support and projection checks for candidates."""

import math
import aiming
from vision_pixels import (
    background_distance,
    bounded_pawn_mask,
    material_projection,
    row_background,
)
from vision_geometry import (
    _shape,
)

def _exposed_upper_edge(contour, image, pawn_origin=None, *, background_like=None):
    """A small oval must border the sky, rather than lie inside another top."""
    import cv2
    import numpy as np
    height, width = image.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(contour), 255)
    # Sample beyond antialiasing, along only the upper half of the perimeter.
    near = max(1, round(width * 0.005))
    far = near + max(1, round(width * 0.005))
    outer = cv2.dilate(mask, np.ones((2 * far + 1, 2 * far + 1), np.uint8))
    inner = cv2.dilate(mask, np.ones((2 * near + 1, 2 * near + 1), np.uint8))
    ring = (outer > 0) & (inner == 0)
    x, y, bw, bh = cv2.boundingRect(contour)
    ring[y + bh // 2:] = False
    if np.count_nonzero(ring) < 20:
        return False
    if pawn_origin is not None:
        # A reconstructed corner may be hidden by the actual pawn. Ignore
        # only its bounded purple silhouette, never an arbitrary obstruction.
        pawn = bounded_pawn_mask(image, pawn_origin)
        hidden = cv2.dilate(pawn, np.ones((3, 3), np.uint8)) > 0
        if np.count_nonzero(ring & hidden) > np.count_nonzero(ring) * 0.25:
            return False
        ring &= ~hidden
        if np.count_nonzero(ring) < 20:
            return False
    # Each row has its own sky color, accommodating vertical gradients.
    if background_like is not None:
        samples = background_like[ring]
    else:
        background = row_background(image)
        rows, columns = np.nonzero(ring)
        samples = np.max(abs(image[rows, columns].astype(float) - background[rows, 0]), axis=1) < 12
    return float(np.mean(samples)) >= 0.85

def _supported_small_surface(contour, image, depth_ratio=0.6):
    """Require a visible pillar/body below a tiny top, not a floating dot."""
    import numpy as np
    import cv2
    height, width = image.shape[:2]
    x, y, bw, bh = cv2.boundingRect(contour)
    cx = x + bw / 2
    y1 = y + bh + max(1, round(width * 0.005))
    y2 = y1 + max(6, round(bw * depth_ratio))
    x1, x2 = round(cx - bw * 0.15), round(cx + bw * 0.15)
    if y2 >= height or x1 < 0 or x2 > width or x2 <= x1:
        return False
    background = row_background(image)
    different = background_distance(image, background) > 8
    # Check each vertical section, so a single trim line does not count as body.
    body = different[y1:y2, x1:x2]
    return all(float(np.mean(part)) >= 0.8 for part in np.array_split(body, 3))

def _platform_candidates(contours, origin, width, angle, detector, image=None, small_surface=False,
                         pawn_origin=None):
    import cv2
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, bw, bh = cv2.boundingRect(contour)
        # Quantized sky gradients can form smooth regions cut by the frame.
        # A landing surface must be complete, with visible space around it.
        if image is not None and (x == 0 or x + bw >= width or y == 0 or y + bh >= image.shape[0]):
            continue
        min_area, min_width = (0.001, 0.05) if small_surface else (0.0025, 0.08)
        if not (min_area * width ** 2 < area < 0.18 * width ** 2 and
                min_width * width < bw < 0.65 * width and 0.3 < bh / bw < 0.85):
            continue
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0 or area / hull_area < 0.85:
            continue
        shape = _shape(contour, width, angle)
        if shape is None:
            continue
        small_top = area <= 0.008 * width ** 2
        if small_surface:
            if (bw > width * 0.16 or bh < width * 0.025 or image is None or
                    not _exposed_upper_edge(contour, image, pawn_origin) or
                    not _supported_small_surface(contour, image)):
                continue
        if not small_surface and shape == "diamond" and (bw <= 0.12 * width or area <= 0.004 * width ** 2):
            continue
        if shape == "ellipse" and (small_top or bw <= 0.12 * width):
            if image is None or not _exposed_upper_edge(contour, image, pawn_origin):
                continue
        moments = cv2.moments(contour)
        center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
        if center[1] > origin[1] - 0.04 * width:
            continue
        horizontal_gap = abs(center[0] - origin[0])
        if horizontal_gap < width * 0.12:
            # A nearby next platform may lie inside the old fixed 48px cutoff.
            # Keep its center outside the pawn's footprint and require an
            # exposed upper boundary, then apply the usual separation/margin
            # checks. This must not admit a pattern on the occupied surface.
            if (horizontal_gap < width * 0.06 or image is None or
                    not _exposed_upper_edge(contour, image, pawn_origin)):
                continue
        if cv2.pointPolygonTest(hull, origin, True) > -0.02 * width:
            continue
        projection = aiming.project_target(origin, center, angle)
        if projection["offset"] > 0.09 * width:
            continue
        margin = cv2.pointPolygonTest(hull, projection["point"], True)
        required_margin = width * (0.02 if small_top else 0.007)
        if small_top and shape == "ellipse":
            required_margin = max(width * 0.014, min(required_margin, bh * 0.25))
        if small_top and shape == "diamond":
            # Fixed eight-pixel clearance consumes most of a narrow top.
            # Scale to its center clearance, retaining a five-pixel floor at
            # capture width 400. Require an exposed rim to reject decoration.
            center_clearance = cv2.pointPolygonTest(hull, center, True)
            scaled_margin = max(width * 0.0125, min(required_margin, center_clearance * 0.4))
            if margin < required_margin:
                if image is None or not _exposed_upper_edge(contour, image, pawn_origin):
                    continue
                required_margin = scaled_margin
        if small_surface:
            # Tiny late-game tops need clearance relative to their own size.
            # Keep a resolution floor; never accept a ray grazing the rim.
            center_clearance = cv2.pointPolygonTest(hull, center, True)
            relative_margin = center_clearance * 0.25
            if shape == "diamond":
                # Slanted pixel edges fluctuate by fractions of a pixel. Allow
                # half a normalized pixel on the size-based requirement only;
                # the absolute 2.4px floor at width 400 remains unchanged.
                relative_margin -= width * 0.00125
            required_margin = max(width * 0.006, min(width * 0.02, relative_margin))
        if margin < max(2, required_margin):
            continue
        candidate = {"center": center, "shape": shape, "area": area,
                     "projection": projection, "landing_margin": margin,
                     "outline": hull.reshape(-1, 2).tolist(), "detector": detector}
        # Inner/outer contours from the same edge must not become two targets.
        duplicate = next((c for c in candidates if math.dist(c["center"], center) < width * 0.025), None)
        if duplicate is None:
            candidates.append(candidate)
        elif area > duplicate["area"]:
            candidates[candidates.index(duplicate)] = candidate
    return candidates

def _trimmed_upper_edge(contour, image):
    """Accept a narrow, same-material rim only when sky lies just beyond it."""
    import cv2
    import numpy as np
    width = image.shape[1]
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, contour, 255)
    interior = cv2.erode(mask, np.ones((3, 3), np.uint8)) > 0
    if np.count_nonzero(interior) < 50:
        return False
    base = np.median(image[interior], axis=0).astype(float)
    if np.dot(base, base) < 100:
        return False
    pixels = image.astype(float)
    brightness, error = material_projection(pixels, base, np.dot(base, base))
    material = (brightness > 0.85) & (brightness < 1.1) & (error < 8)
    background = row_background(image)
    sky = np.max(abs(pixels - background), axis=2) < 12
    # Antialiased rim/background boundaries contain blends of both colors.
    delta = background - base
    denominator = np.sum(delta * delta, axis=2)
    blend = np.sum((pixels - base) * delta, axis=2) / np.maximum(denominator, 1)
    blended = base + blend[:, :, None] * delta
    transition = ((denominator > 100) & (blend >= 0) & (blend <= 1) &
                  (np.max(abs(pixels - blended), axis=2) < 8))
    near = max(2, round(width * 0.0125))
    far = near + max(1, round(width * 0.005))
    inner = cv2.dilate(mask, np.ones((2 * near + 1, 2 * near + 1), np.uint8))
    outer = cv2.dilate(mask, np.ones((2 * far + 1, 2 * far + 1), np.uint8))
    band = (inner > 0) & (mask == 0)
    ring = (outer > 0) & (inner == 0)
    _, y, _, bh = cv2.boundingRect(contour)
    band[y + bh // 2:] = False
    ring[y + bh // 2:] = False
    return (np.count_nonzero(band) >= 20 and np.count_nonzero(ring) >= 20 and
            np.mean((sky | material | transition)[band]) >= 0.97 and np.mean(sky[ring]) >= 0.85)
