"""Candidate tops supported by observed slanted edges and material evidence."""

import math
from vision_pixels import (
    channel_edges,
    background_distance,
    material_projection,
    row_background,
)
from vision_geometry import (
    diamond_vertices,
    edge_samples,
    slanted_line_groups,
)
from vision_validation import (
    _exposed_upper_edge,
    _platform_candidates,
)
from vision_materials import (
    _printed_surface,
    _striped_surface,
)

def _accumulated_slanted_lines(edges, angle, min_support):
    """Line intercept proposals from all pixels, restricted to the two axes."""
    import cv2
    import numpy as np
    groups = {1: [], -1: []}
    for sign in groups:
        theta = math.pi / 2 + sign * math.radians(angle)
        # A single angular bin keeps the accumulator aligned with the same
        # projection geometry used to intersect and validate the four sides.
        lines = cv2.HoughLines(edges, 1, np.pi / 180, min_support,
                               min_theta=theta, max_theta=theta + 1e-4)
        if lines is not None:
            groups[sign] = [float(rho / np.sin(t)) for rho, t in lines[:, 0]]
    return groups


def _marked_top_contours(image, origin, angle, color_edges=False, partial_edge=False,
                         printed_texture=False, small_surface=False):
    """Recover marked tops from four edges, or three with partial fourth-edge evidence."""
    import cv2
    import numpy as np
    from itertools import combinations
    height, width = image.shape[:2]
    bright = ((image.min(axis=2) >= 220) &
              (image.max(axis=2).astype(int) - image.min(axis=2) < 20)).astype(np.uint8) * 255
    bright[:round(height * 0.22)] = 0
    bright[round(origin[1] - width * 0.04):] = 0
    blobs, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    markers = []
    for blob in blobs:
        area = cv2.contourArea(blob)
        x, y, bw, bh = cv2.boundingRect(blob)
        if not (width ** 2 * 0.00005 < area < width ** 2 * 0.0015 and
                width * 0.008 < bw < width * 0.07 and 0.35 < bh / bw < 0.85):
            continue
        moments = cv2.moments(blob)
        markers.append((moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]))
    if not markers:
        return []
    proposal_scale = 0.5 if small_surface else 1.0
    gray = cv2.cvtColor(cv2.GaussianBlur(image, (3, 3) if small_surface else (5, 5), 0), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 15, 30)
    if color_edges:
        # Different colors can share luminance, hiding a real boundary in gray.
        edges |= channel_edges(image)
    slope = math.tan(math.radians(angle))
    if printed_texture:
        # Busy print fragments probabilistic line segments: even one changed
        # edge pixel can alter their selected extents. Accumulate all edge
        # pixels at the known top-edge angles instead. These are proposals;
        # each finite side still needs the pixel-support checks below.
        groups = _accumulated_slanted_lines(edges, angle, round(width * 0.0625))
    else:
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, round(20 * proposal_scale),
                              minLineLength=round(width * 0.0625 * proposal_scale),
                              maxLineGap=round(width * 0.015))
        if lines is None:
            return []
        groups = slanted_line_groups(lines, angle, 6)
    distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    background = row_background(image)
    background_like = background_distance(image, background) < 12
    contours = []
    for mx, my in markers:
        pairs = {}
        for sign in (1, -1):
            middle = my - sign * slope * mx
            nearby = sorted(b for b in groups[sign] if abs(b - middle) < width * 0.21)
            pairs[sign] = [(lo, hi) for lo, hi in combinations(nearby, 2)
                           if lo < middle < hi and width * 0.07 * proposal_scale < hi - lo < width * (0.09 if small_surface else 0.4)]
            if partial_edge:
                # Marker symmetry proposes the opposite edge, whose visible
                # part still needs independent support below.
                for edge in nearby:
                    lo, hi = sorted((edge, 2 * middle - edge))
                    if width * 0.07 * proposal_scale < hi - lo < width * (0.09 if small_surface else 0.4):
                        pairs[sign].append((lo, hi))
        for low_a, high_a in pairs[1]:
            for low_b, high_b in pairs[-1]:
                if abs((high_a - low_a) - (high_b - low_b)) > width * 0.02 * proposal_scale:
                    continue
                cx = (low_b + high_b - low_a - high_a) / (4 * slope)
                cy = (low_a + high_a + low_b + high_b) / 4
                if math.dist((cx, cy), (mx, my)) > width * 0.015 * proposal_scale:
                    continue
                polygon = diamond_vertices(low_a, high_a, low_b, high_b, slope)
                if polygon[:, 0].min() <= 2 or polygon[:, 0].max() >= width - 2 or polygon[:, 1].min() <= 2 or polygon[:, 1].max() >= height - 2:
                    continue
                supported = True
                strong_edges = 0
                for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
                    samples = edge_samples(a, b, 0.05, 0.95)
                    xy = np.rint(samples).astype(int)
                    errors = distance[xy[:, 1], xy[:, 0]]
                    if np.mean(errors <= 2) >= 0.75 and np.mean(errors) <= 1.5:
                        strong_edges += 1
                    elif not partial_edge or np.mean(errors <= 2) < 0.35 or np.mean(errors) > 5:
                        supported = False
                        break
                if supported and (not partial_edge or strong_edges >= 3):
                    contour = np.rint(polygon).astype(np.int32).reshape(-1, 1, 2)
                    # A print stroke (or the inside of a damaged rim) can also
                    # accumulate into a line. Require the upper boundary to
                    # face the background, rather than another part of the top.
                    if (partial_edge or printed_texture) and not _exposed_upper_edge(
                            contour, image, background_like=background_like):
                        continue
                    inside = np.zeros(image.shape[:2], np.uint8)
                    cv2.fillConvexPoly(inside, contour, 255)
                    inside = cv2.erode(inside, np.ones((3, 3), np.uint8)) > 0
                    # Four surrounding lines do not prove that a missing corner
                    # or hole is solid. Reject visible sky within the surface.
                    if not np.any(inside):
                        continue
                    if partial_edge:
                        # Pale grain can resemble the sky in isolated pixels.
                        # Reject coherent sky patches instead; even a small
                        # hole with a 3x3 interior is not evidence of solid wood.
                        core = cv2.erode((background_like & inside).astype(np.uint8),
                                         np.ones((3, 3), np.uint8))
                        if np.any(core):
                            continue
                    elif np.mean(background_like[inside]) >= 0.02:
                        continue
                    pixels = image[inside].astype(float)
                    pixels = pixels[pixels.min(axis=1) < 220]  # Exclude the white marker.
                    if len(pixels) < 50:
                        continue
                    base = np.median(pixels, axis=0)
                    norm = float(np.dot(base, base))
                    if norm < 100:
                        continue
                    brightness, residual = material_projection(pixels, base, norm)
                    # Grain may vary in brightness, but an unrelated material in
                    # the inferred surface is not evidence of a complete top.
                    if (np.mean(residual < 12) >= 0.97 or _striped_surface(image, inside, base, angle) or
                            (printed_texture and not partial_edge and strong_edges == 4 and
                             _printed_surface(image, inside, base))):
                        contours.append(contour)
    return contours

def _unmarked_textured_top_contours(image, origin, angle):
    """Recover a textured top from four observed edges, without a center dot."""
    import cv2
    import numpy as np
    from itertools import combinations
    height, width = image.shape[:2]
    smooth = cv2.bilateralFilter(image, 9, 60, width * 0.0375)
    edges = np.zeros((height, width), np.uint8)
    for channel in cv2.split(smooth):
        edges |= cv2.Canny(channel, 10, 25)
    edges[:round(height * 0.22)] = 0
    edges[round(origin[1]):] = 0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 18,
                            minLineLength=max(12, round(width * 0.06)), maxLineGap=5)
    if lines is None:
        return []
    slope = math.tan(math.radians(angle))
    groups = slanted_line_groups(lines, angle, 4)
    pairs = {}
    for sign, values in groups.items():
        # Merge duplicate observations of a line before forming quadrilaterals.
        merged = []
        for value in sorted(values):
            if not merged or value-merged[-1] > 2:
                merged.append(value)
        # An excessively busy scene offers no reliable line-only fallback.
        if len(merged) > 50:
            return []
        pairs[sign] = [(lo, hi) for lo, hi in combinations(merged, 2)
                       if width * 0.10 < hi-lo < width * 0.32]
    distance = cv2.distanceTransform(255-edges, cv2.DIST_L2, 3)
    background = row_background(image)
    background_like = background_distance(image, background) < 12
    contours = []
    for low_a, high_a in pairs[1]:
        for low_b, high_b in pairs[-1]:
            if abs((high_a-low_a)-(high_b-low_b)) > 3:
                continue
            polygon = diamond_vertices(low_a, high_a, low_b, high_b, slope)
            if (polygon[:,0].min() < 6 or polygon[:,0].max() >= width-6 or
                    polygon[:,1].min() < height*.22 or polygon[:,1].max() >= origin[1]):
                continue
            contour = np.rint(polygon).astype(np.int32).reshape(-1,1,2)
            if not _platform_candidates([contour], origin, width, angle, "four-observed-edges", image):
                continue
            if not _exposed_upper_edge(contour, image):
                continue
            supported = True
            for a, b in zip(polygon, np.roll(polygon,-1,axis=0)):
                samples = edge_samples(a, b, .08, .92)
                xy = np.rint(samples).astype(int)
                errors = distance[xy[:,1],xy[:,0]]
                if np.mean(errors <= 2) < .85 or np.mean(errors) > 1.2:
                    supported = False
                    break
                if (a[1]+b[1])/2 > polygon[:,1].mean():
                    # Front edges must separate the top from a darker solid
                    # side wall. Texture strokes and ground shadows do not.
                    inner = np.rint(samples + (0,-4)).astype(int)
                    outer = np.rint(samples + (0,4)).astype(int)
                    delta = smooth[inner[:,1],inner[:,0]].astype(float) - smooth[outer[:,1],outer[:,0]]
                    if np.median(delta.mean(axis=1)) < 8 or np.mean(delta.mean(axis=1)>4) < .8:
                        supported = False
                        break
            if not supported:
                continue
            mask = np.zeros((height,width),np.uint8)
            cv2.fillConvexPoly(mask,contour,255)
            inside = cv2.erode(mask,np.ones((5,5),np.uint8)) > 0
            if not np.any(inside) or np.mean(background_like[inside]) >= .01:
                continue
            pixels = image[inside].astype(float)
            base = np.median(pixels,axis=0)
            norm = float(base@base)
            if norm < 100:
                continue
            brightness, residual = material_projection(pixels, base, norm)
            # Four observed outer edges may enclose several color regions.
            # Reuse the continuous, aligned decoration check; an arbitrary
            # foreign patch still cannot complete an otherwise invalid top.
            if np.mean(residual<12) < .97 and not _striped_surface(image, inside, base, angle):
                continue
            contours.append(contour)
    return contours
