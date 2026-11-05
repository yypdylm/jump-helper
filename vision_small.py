"""Small and shaded top recovery, with independent boundary/support checks."""

from vision_pixels import (
    _color_contours,
    background_distance,
    blend_projection,
    material_projection,
    row_background,
)
from vision_geometry import (
    _shape,
)
from vision_validation import (
    _exposed_upper_edge,
    _supported_small_surface,
)

def _shadow_ellipse_contours(image, angle, *, small_surface=True):
    """Recover a shaded oval with geometry, material and physical support evidence."""
    import cv2
    import numpy as np
    height, width = image.shape[:2]
    background = row_background(image)
    background_error = background_distance(image, background)
    background_like = background_error < 12
    result = []
    for contour in _color_contours(image, 12, False, min_area_ratio=0.001):
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        # Use the same diameter limit as the small-surface validator. An
        # intact 16%-wide oval can exceed the old 0.8% area cap even when a
        # shadow removes part of its color region.
        bw = cv2.boundingRect(hull)[2]
        min_width, max_width = (0.05, 0.16) if small_surface else (0.16, 0.65)
        max_area = 0.012 if small_surface else 0.18
        if not (width ** 2 * 0.001 < area < width ** 2 * max_area and
                width * min_width < bw <= width * max_width) or hull_area <= 0:
            continue
        if not 0.6 < area / hull_area < 0.95 or _shape(hull, width, angle) != "ellipse":
            continue
        (cx, cy), (a, b), degrees = cv2.fitEllipse(hull)
        ellipse = cv2.ellipse2Poly((round(cx), round(cy)), (round(a / 2), round(b / 2)),
                                  round(degrees), 0, 360, 10).reshape(-1, 1, 2)
        # Larger tops reuse the same recovery, but must also expose sky and
        # a supporting body. Keep tiny tops on their stricter validation path.
        if not small_surface and (cv2.boundingRect(ellipse)[2] <= width * 0.16 or
                not _exposed_upper_edge(ellipse, image) or
                not _supported_small_surface(ellipse, image)):
            continue
        surface = np.zeros((height, width), np.uint8)
        visible = surface.copy()
        cv2.fillConvexPoly(surface, ellipse, 255)
        cv2.fillPoly(visible, [contour], 255)
        if np.count_nonzero((visible > 0) & (surface > 0)) / max(1, np.count_nonzero(visible)) < 0.95:
            continue
        # Ignore the antialiased perimeter; verify the actual shaded interior.
        interior = cv2.erode(surface, np.ones((3, 3), np.uint8)) > 0
        missing = interior & (visible == 0)
        if np.count_nonzero(missing) < 20:
            continue
        base = np.median(image[visible > 0], axis=0).astype(float)
        norm = float(np.dot(base, base))
        if norm < 100:
            continue
        pixels = image[missing].astype(float)
        shade, residual = material_projection(pixels, base, norm)
        # A shaded material can resemble the sky at its antialiased boundary.
        # Count it as sky only when the measured background explains it at
        # least as well as the top's pigment. Exact sky and ties still reject.
        sky = background_like[missing] & (background_error[missing] <= residual)
        if np.mean(sky) > 0.02:
            continue
        if np.mean((shade > 0.45) & (shade < 0.98) & (residual < 12)) >= 0.9:
            result.append(ellipse)
    return result

def _two_tone_small_ellipse_contours(image, angle):
    """Join a tiny top and its shadow using both observed outer arcs."""
    import cv2
    import numpy as np
    from itertools import combinations
    width = image.shape[1]
    parts = []
    for contour in _color_contours(image, 12, False, min_area_ratio=0.001):
        area = cv2.contourArea(contour)
        x, y, bw, bh = cv2.boundingRect(contour)
        if not (width ** 2 * 0.0005 < area < width ** 2 * 0.008 and
                bw < width * 0.16 and bh < width * 0.12):
            continue
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        parts.append((contour, mask, np.median(image[mask > 0], axis=0)))
    background = row_background(image)
    background_like = background_distance(image, background) < 12
    result = []
    for (a, ma, ca), (b, mb, cb) in combinations(parts, 2):
        hull = cv2.convexHull(np.vstack([a, b]))
        x, y, bw, bh = cv2.boundingRect(hull)
        area = cv2.contourArea(hull)
        if not (width * 0.05 < bw < width * 0.16 and
                width ** 2 * 0.001 < area < width ** 2 * 0.008):
            continue
        base, dark = (ca, cb) if np.linalg.norm(ca) > np.linalg.norm(cb) else (cb, ca)
        norm = float(np.dot(base, base))
        if norm < 100:
            continue
        shade = float(dark @ base / norm)
        if not (0.45 < shade < 0.98 and np.max(abs(dark - shade * base)) < 12):
            continue
        if (_shape(hull, width, angle) != "ellipse" or
                not _exposed_upper_edge(hull, image) or
                not _supported_small_surface(hull, image)):
            continue
        # Adjacent regions must supply most of the observed oval; no remote
        # patches, fitted unseen arc, or sky gap may complete its interior.
        if not np.any((cv2.dilate(ma, np.ones((3, 3), np.uint8)) > 0) & (mb > 0)):
            continue
        filled = np.zeros(image.shape[:2], np.uint8)
        cv2.fillConvexPoly(filled, hull, 255)
        inside = cv2.erode(filled, np.ones((3, 3), np.uint8)) > 0
        union = (ma > 0) | (mb > 0)
        if (np.mean(union[filled > 0]) < 0.75 or
                np.mean(background_like[inside]) >= 0.02):
            continue
        pixels = image[inside].astype(float)
        brightness, residual = material_projection(pixels, base, norm)
        if np.mean((brightness > 0.45) & (brightness < 1.05) & (residual < 12)) >= 0.97:
            result.append(hull)
    return result

def _shadow_small_diamond_contours(image, angle):
    return _shadow_marked_top_contours(image, angle, "diamond")

def _shadow_marked_top_contours(image, angle, shape, tiny_marker=False):
    """Validate a shaded top's visible hull using its material and center mark."""
    import cv2
    import numpy as np
    width = image.shape[1]
    result = []
    for contour in _color_contours(image, 12, False, min_area_ratio=0.001):
        hull = cv2.convexHull(contour)
        area, full_area = cv2.contourArea(contour), cv2.contourArea(hull)
        x, y, bw, bh = cv2.boundingRect(hull)
        if tiny_marker and (shape != "ellipse" or not width * 0.05 < bw <= width * 0.12):
            continue
        max_area = 0.03 if shape == "ellipse" else 0.008
        if not width ** 2 * 0.001 < full_area < width ** 2 * max_area:
            continue
        min_fill = 0.6 if shape == "ellipse" and not tiny_marker else 0.5
        if not min_fill < area / full_area < 0.98 or _shape(hull, width, angle) != shape:
            continue
        if shape == "ellipse" and (not _exposed_upper_edge(hull, image) or
                                   not _supported_small_surface(hull, image)):
            continue
        region = np.zeros(image.shape[:2], np.uint8)
        filled = region.copy()
        cv2.fillPoly(region, [contour], 255)
        cv2.fillConvexPoly(filled, hull, 255)
        interior = cv2.erode(filled, np.ones((3, 3), np.uint8)) > 0
        missing = interior & (region == 0)
        if np.count_nonzero(missing) < 20:
            continue
        base = np.median(image[region > 0], axis=0)
        if np.dot(base, base) < 100:
            continue
        pixels = image.astype(float)
        brightness, residual = material_projection(pixels, base, np.dot(base, base))
        shadow = (brightness > 0.45) & (brightness < 0.98) & (residual < 12)
        white = (pixels.min(axis=2) >= 220) & (np.ptp(pixels, axis=2) < 15) & interior
        marks, _ = cv2.findContours(white.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, bw, bh = cv2.boundingRect(hull)
        marker = np.zeros(image.shape[:2], np.uint8)
        for mark in marks:
            mx, my, mw, mh = cv2.boundingRect(mark)
            if (cv2.contourArea(mark) < 5 or mw > bw * (0.45 if tiny_marker else 0.35) or mh > bh * 0.4 or
                    not 0.35 < mh / mw < 0.8 or
                    abs(mx + mw / 2 - x - bw / 2) > bw * 0.15 or
                    abs(my + mh / 2 - y - bh / 2) > bh * 0.15 or
                    not 0.55 < cv2.contourArea(mark) / (mw * mh) < 0.9):
                continue
            cv2.drawContours(marker, [mark], -1, 255, -1)
        # Only the narrow antialiased rim around a validated central oval may
        # be brighter than the material. A white hole elsewhere cannot pass.
        core = marker > 0
        if np.any(core):
            # A white marker changes hue on a colored top. Its edge blends
            # white with the material, rather than simply brightening it.
            white_color = np.median(pixels[core], axis=0)
            delta = white_color - base
            norm = np.dot(delta, delta)
            if norm > 100:
                blend, error = blend_projection(pixels, base, delta, norm)
                rim = cv2.dilate(marker, np.ones((3, 3), np.uint8)) > 0
                core |= rim & (blend >= 0) & (blend <= 1.1) & (error < 12)
        marker = core
        if shape == "ellipse" and not np.any(marker):
            continue
        shadow_evidence = np.mean(shadow[missing]) >= 0.45
        if tiny_marker:
            # The white center dot occupies more of a tiny late-game top.
            # Measure the shaded material separately from that validated dot,
            # rather than requiring shadow to dominate dot + missing material.
            shaded_material = missing & ~marker
            shadow_evidence = (np.count_nonzero(shaded_material) >= 20 and
                               np.mean(shadow[shaded_material]) >= 0.90)
        if shadow_evidence and np.mean((shadow | marker)[missing]) >= 0.95:
            result.append(hull)
    return result
