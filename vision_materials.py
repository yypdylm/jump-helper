"""Candidate tops supported by connected colors, seams, shadows or rings."""

import math
from vision_pixels import (
    _color_contours,
    background_distance,
    blend_projection,
    color_codes,
    material_projection,
    row_background,
)
from vision_geometry import (
    _shape,
    edge_angles,
    narrow_seam,
)
from vision_validation import (
    _exposed_upper_edge,
    _supported_small_surface,
)

def _grid_top_contours(image, bin_width, angle):
    """Join same-color tiles only across narrow, visibly white grid seams."""
    import cv2
    import numpy as np
    height, width = image.shape[:2]
    codes = color_codes(image, bin_width)
    size = max(3, round(width * 0.0125)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    background = row_background(image)
    background_like = background_distance(image, background) < 12
    result = []
    for code, count in zip(*np.unique(codes, return_counts=True)):
        if count < width ** 2 * 0.004:
            continue
        mask = (codes == code).astype(np.uint8) * 255
        mask[:round(height * 0.22)] = 0
        mask[round(height * 0.87):] = 0
        joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if not width ** 2 * 0.004 < cv2.contourArea(contour) < width ** 2 * 0.18:
                continue
            if _shape(contour, width, angle) != "diamond":
                continue
            region = np.zeros_like(mask)
            cv2.drawContours(region, [contour], -1, 255, -1)
            tiles = cv2.bitwise_and(mask, region)
            _, _, stats, _ = cv2.connectedComponentsWithStats(tiles)
            if np.count_nonzero(stats[1:, cv2.CC_STAT_AREA] >= width ** 2 * 0.0005) < 4:
                continue
            added = (region > 0) & (joined > 0) & (mask == 0)
            if np.count_nonzero(added) < 30 or np.mean(background_like[added]) > 0.02:
                continue
            pixels = image[added].astype(float)
            base = np.median(image[tiles > 0], axis=0)
            to_white = 255 - base
            norm = float(np.dot(to_white, to_white))
            if norm < 100:
                continue
            alpha, error = blend_projection(pixels, base, to_white, norm)
            blended_white = (alpha >= -0.05) & (alpha <= 1.05) & (error < 15)
            white_core = (pixels.min(axis=1) >= 220) & (np.ptp(pixels, axis=1) < 25)
            if np.mean(blended_white) >= 0.95 and np.mean(white_core) >= 0.45:
                result.append(contour)
    return result

def _striped_surface(image, inside, base, angle):
    """Allow one narrow, coherent stripe across an otherwise uniform top."""
    import cv2
    import numpy as np
    # Only surface pixels contribute to this check. Keep the full-size mask
    # for connected-component geometry without projecting the whole frame.
    _, residual = material_projection(image[inside].astype(float), base)
    stripe = np.zeros(inside.shape, np.uint8)
    stripe[inside] = (residual >= 12).astype(np.uint8)
    count = int(np.count_nonzero(stripe))
    fraction = count / max(1, np.count_nonzero(inside))
    if not 0.04 <= fraction <= 0.25:
        return False
    n, labels, stats, centers = cv2.connectedComponentsWithStats(stripe)
    if n < 2:
        return False
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[index, cv2.CC_STAT_AREA] < count * 0.9:
        return False
    component = (labels == index).astype(np.uint8)
    ys, xs = np.where(component)
    rectangle = cv2.minAreaRect(np.column_stack((xs, ys)).astype(np.float32))
    center = rectangle[0]
    sy, sx = np.where(inside)
    surface_width = int(sx.max() - sx.min() + 1)
    if not narrow_seam(rectangle, count, surface_width, angle):
        return False
    if math.dist(center, (float(np.mean(sx)), float(np.mean(sy)))) > image.shape[1] * 0.02:
        return False
    # Exclude blended edge pixels and the white center marker when checking
    # that the stripe itself is a real, consistent color region.
    core = cv2.erode(component, np.ones((3, 3), np.uint8)) > 0
    pixels = image[core].astype(float)
    pixels = pixels[pixels.min(axis=1) < 220]
    if len(pixels) < 30:
        return False
    stripe_color = np.median(pixels, axis=0)
    norm = float(np.dot(stripe_color, stripe_color))
    if norm < 100:
        return False
    _, error = material_projection(pixels, stripe_color, norm)
    return float(np.mean(error < 15)) >= 0.95

def _printed_surface(image, inside, base):
    """Require a repeated second pigment, not an arbitrary mixed-color region.

    Called only after a centered marker and four fully supported outer edges.
    Both pigments retain the single-material residual and coverage requirements.
    Disconnected, distributed ink patches distinguish printing from one foreign
    surface or a large occluder. All dimensions are relative to the visible top.
    """
    import cv2
    import numpy as np
    visible = inside & (image.min(axis=2) < 220)
    pixels = image[visible].astype(float)
    if len(pixels) < 50:
        return False
    def residual(color):
        norm = float(color @ color)
        if norm < 100:
            return np.full(len(pixels), np.inf)
        return material_projection(pixels, color, norm)[1]
    base_error = residual(base)
    outliers = base_error >= 12
    fraction = float(np.mean(outliers))
    if not 0.05 <= fraction <= 0.30:
        return False
    ink = np.median(pixels[outliers], axis=0)
    ink_error = residual(ink)
    if np.mean((base_error < 12) | (ink_error < 12)) < 0.97:
        return False
    mask = np.zeros(inside.shape, np.uint8)
    mask[visible] = (outliers & (ink_error < 12)).astype(np.uint8)
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    surface_area = len(pixels)
    if count < 4 or np.max(areas) > surface_area * 0.08:
        return False
    patches = centers[1:][areas >= max(3, surface_area * 0.003)]
    if len(patches) < 3:
        return False
    ys, xs = np.nonzero(inside)
    span = np.array([np.ptp(xs), np.ptp(ys)], float)
    return bool(np.all(np.ptp(patches, axis=0) >= span * 0.35))

def _taped_top_contours(image, angle, bin_width=12):
    """Join two matching top halves only across a solid, narrow material seam."""
    import cv2
    import numpy as np
    from itertools import combinations
    width = image.shape[1]
    parts = []
    for contour in _color_contours(image, bin_width, False):
        area = cv2.contourArea(contour)
        if not width ** 2 * 0.003 < area < width ** 2 * 0.08:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        polygon = cv2.approxPolyDP(contour, cv2.arcLength(contour, True) * 0.025, True).reshape(-1, 2)
        if area / max(1, hull_area) < 0.95 or len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        slopes = edge_angles(polygon)
        if any(abs(slope-angle) > 13 for slope in slopes):
            continue
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        parts.append((contour, area, mask, np.median(image[mask > 0], axis=0)))
    background = row_background(image)
    result = []
    for (a, aa, ma, ca), (b, ab, mb, cb) in combinations(parts, 2):
        if np.max(abs(ca-cb)) > 8 or np.any((ma > 0) & (mb > 0)):
            continue
        hull = cv2.convexHull(np.vstack([a,b]))
        area = cv2.contourArea(hull)
        if (not 0.65 < (aa+ab)/area < 0.95 or min(aa,ab)/area < 0.25 or
                _shape(hull,width,angle) != "diamond" or not _exposed_upper_edge(hull,image)):
            continue
        filled = np.zeros(image.shape[:2], np.uint8)
        cv2.fillConvexPoly(filled, hull, 255)
        inside = cv2.erode(filled,np.ones((3,3),np.uint8)) > 0
        gap = inside & (ma == 0) & (mb == 0)
        yy,xx = np.where(gap)
        if len(xx) < 30:
            continue
        rectangle = cv2.minAreaRect(np.column_stack([xx, yy]).astype(np.float32))
        if not narrow_seam(rectangle, len(xx), cv2.boundingRect(hull)[2], angle):
            continue
        # Geometric completion alone must not turn a sky gap into solid tape.
        if np.mean(background_distance(image, background)[inside] < 12) >= 0.02:
            continue
        base = (ca+cb)/2
        if np.dot(base,base) < 100:
            continue
        pixels = image[gap].astype(float)
        brightness, residual = material_projection(pixels, base, np.dot(base,base))
        if np.mean((brightness > 0.6) & (brightness < 1.02) & (residual < 12)) >= 0.97:
            result.append(hull)
    return result

def _shadow_top_contours(image, bin_width, angle):
    """Repair small concavities only when their pixels support a cast shadow."""
    import cv2
    import numpy as np
    width = image.shape[1]
    result = []
    for contour in _color_contours(image, bin_width):
        area = cv2.contourArea(contour)
        if not 0.004 * width ** 2 < area < 0.18 * width ** 2:
            continue
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if not 0.84 <= area / hull_area <= 0.98 or _shape(hull, width, angle) != "diamond":
            continue
        region = np.zeros(image.shape[:2], np.uint8)
        filled = region.copy()
        cv2.fillPoly(region, [contour], 255)
        cv2.fillConvexPoly(filled, hull, 255)
        # Hull rounding and antialiasing add a thin rim of unrelated pixels.
        # Test the actual concavity interior, not that outer transition rim.
        interior = cv2.erode(filled, np.ones((3, 3), np.uint8))
        missing = (interior > 0) & (region == 0)
        if np.count_nonzero(missing) < 20:
            continue
        base = np.median(image[region > 0], axis=0)
        if float(np.dot(base, base)) < 100:
            continue
        pixels = image[missing].astype(float)
        brightness, residual = material_projection(pixels, base, np.dot(base, base))
        # A hole, another platform or the sky cannot be filled on shape alone.
        shadow = (brightness > 0.45) & (brightness < 0.98) & (residual < 12)
        if float(np.mean(shadow)) >= 0.90:
            result.append(hull)
    return result

def _concentric_top_contours(image, angle):
    """Recover interrupted rings only with concentric geometry and solid fill."""
    import cv2
    import numpy as np
    width = image.shape[1]
    rings = []
    for contour in _color_contours(image, 12, False, min_area_ratio=0.001):
        hull = cv2.convexHull(contour)
        area = cv2.contourArea(hull)
        if not width ** 2 * 0.002 < area < width ** 2 * 0.03:
            continue
        if not 0.15 < cv2.contourArea(contour)/area < 0.9 or _shape(hull,width,angle) != "ellipse":
            continue
        center, axes, _ = cv2.fitEllipse(hull)
        rings.append((contour,hull,np.array(center),np.sort(axes)))
    background = row_background(image)
    result = []
    for contour,hull,center,axes in rings:
        if not _exposed_upper_edge(hull,image) or not _supported_small_surface(hull,image):
            continue
        scales = []
        for _,_,other_center,other_axes in rings:
            ratio = other_axes/axes
            if (math.dist(center,other_center) < width*0.0075 and
                    np.all((ratio > 0.4) & (ratio < 0.9)) and abs(ratio[0]-ratio[1]) < 0.06):
                scales.append(float(ratio[1]))
        # Require two distinct inner rings, not duplicate quantized edges.
        if len(scales) < 2 or max(scales)-min(scales) < 0.1:
            continue
        region=np.zeros(image.shape[:2],np.uint8);filled=region.copy()
        cv2.fillPoly(region,[contour],255);cv2.fillConvexPoly(filled,hull,255)
        inside=cv2.erode(filled,np.ones((3,3),np.uint8))>0
        if np.mean(background_distance(image, background)[inside] < 12) >= 0.02:
            continue
        base=np.median(image[region>0],axis=0).astype(float)
        if np.dot(base,base)<100:
            continue
        pixels=image[inside].astype(float)
        brightness, residual = material_projection(pixels, base)
        material=(brightness>.45)&(brightness<1.35)&(residual<12)
        # A small central white marker may blend into the ring color. This
        # exception is spatially bounded; unrelated white regions cannot fill
        # a missing rim or hide background inside the inferred surface.
        yy,xx=np.where(inside)
        central=((xx-center[0])/(axes[1]/2))**2+((yy-center[1])/(axes[0]/2))**2 < .3**2
        to_white=245-base;norm=np.dot(to_white,to_white)
        if norm>100:
            blend, error = blend_projection(pixels, base, to_white, norm)
            material |= central & (blend>=0) & (blend<=1.05) & (error<12)
        if np.mean(material)>=.97:
            result.append(hull)
    return result
