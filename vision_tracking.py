"""Scene readiness and camera motion evidence; never sends input."""

import math
from vision_pixels import (
    background_distance,
    restrict_tracking_region,
    row_background,
    tracking_mask,
)

def _stability_masks(image, scene):
    """Only the pawn and landing surface determine readiness, not the sky."""
    import cv2
    import numpy as np
    height, width = image.shape[:2]
    top = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(top, np.array(scene["platform"]["outline"], np.int32), 255)
    player = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), (110, 50, 30), (170, 255, 160))
    x, y = scene["origin"]
    region = np.zeros_like(player)
    cv2.rectangle(region, (round(x - width * 0.065), round(y - width * 0.24)),
                  (round(x + width * 0.065), round(y + width * 0.01)), 255, -1)
    player &= region
    # Ignore antialiasing pixels blending with the changing background.
    kernel = np.ones((3, 3), np.uint8)
    return tuple(cv2.erode(mask, kernel) > 0 for mask in (player, top))

def scenes_stable(first_image, first, second_image, second):
    import cv2
    import numpy as np
    if first_image.shape != second_image.shape:
        return False
    width = second_image.shape[1]
    if math.dist(first.get("aim_origin",first["origin"]),
                 second.get("aim_origin",second["origin"])) > width*0.006:
        return False
    if any(math.dist(first[k], second[k]) > width * 0.006 for k in ("origin", "target")):
        return False
    if first["platform"]["shape"] != second["platform"]["shape"]:
        return False
    diff = cv2.absdiff(first_image, second_image)
    for index, (a, b) in enumerate(zip(_stability_masks(first_image, first), _stability_masks(second_image, second))):
        union = a | b
        count = int(np.count_nonzero(union))
        if count < 50 or np.count_nonzero(a & b) / count < 0.95:
            return False
        pixels = diff[union]
        if float(np.mean(pixels)) >= 2.0 or float(np.mean(np.max(pixels, axis=1) > 20)) >= 0.025:
            if index == 0:
                return False
            # A logo may animate while the landing surface stays still. Only
            # small changes deep inside BOTH outlines may be ignored; keep
            # the pawn, perimeter and broad surface changes under strict checks.
            depth_a = cv2.distanceTransform(a.astype(np.uint8), cv2.DIST_L2, 5)
            depth_b = cv2.distanceTransform(b.astype(np.uint8), cv2.DIST_L2, 5)
            interior = (depth_a > depth_a.max() * 0.6) & (depth_b > depth_b.max() * 0.6)
            rim = union & ~interior
            changed = np.max(diff, axis=2) > 20
            if (np.count_nonzero(changed & union) / count > 0.15 or
                    float(np.mean(diff[rim])) >= 2.0 or
                    float(np.mean(changed[rim])) >= 0.025):
                return False
    return True

def _score_and_player_changed(previous_image, previous_scene, image, scene):
    """Reject color transitions and stationary bonus animations as progress."""
    if math.dist(previous_scene["origin"], scene["origin"]) < image.shape[1] * 0.02:
        return False
    return _score_changed(previous_image, image)

def _score_changed(previous_image, image):
    """Compare solid score strokes independently of camera recentering."""
    import cv2
    import numpy as np
    height, width = image.shape[:2]
    masks = []
    for frame in (previous_image, image):
        roi = frame[round(height * 0.09):round(height * 0.22),
                    round(width * 0.04):round(width * 0.65)].astype(float)
        background = row_background(roi)
        mask = np.min(background - roi, axis=2) > 50
        count = np.count_nonzero(mask)
        if not width ** 2 * 0.001 < count < width ** 2 * 0.03:
            return False
        yy, xx = np.where(mask)
        if not height * 0.025 < np.ptp(yy) < height * 0.06:
            return False
        masks.append(mask)
    # Binarization removes sky color and text antialiasing variations.
    difference = masks[0] ^ masks[1]
    if np.count_nonzero(difference) > width ** 2 * 0.0005:
        return True
    # A one-point increase such as 5122 -> 5123 changes only two short
    # strokes. Require solid cores in both added and removed ink instead of
    # lowering the aggregate threshold to include antialiasing or speckles.
    kernel = np.ones((3, 3), np.uint8)
    for change in (masks[0] & ~masks[1], masks[1] & ~masks[0]):
        core = cv2.erode(change.astype(np.uint8), kernel)
        _, _, stats, _ = cv2.connectedComponentsWithStats(core)
        if len(stats) < 2 or stats[1:, cv2.CC_STAT_AREA].max() < max(4, width ** 2 * 0.00005):
            return False
    return True

def _glowing_material_alignment(previous_image, previous_scene, image, scene,
                                delta, points, group):
    """Verify the pink patterned cube when glow reverses texture contrast.

    Feature matches propose motion; the material silhouette and interior
    gradient magnitudes must independently agree. This does not require old
    neighbors to remain on screen, or the pawn to move in screen coordinates.
    """
    import cv2
    import numpy as np
    height, width = image.shape[:2]
    support = points[group]
    if (len(support) < 10 or len(np.unique(np.round(support / 4), axis=0)) < 8 or
            np.min(np.ptp(support, axis=0)) < width * 0.12 or
            np.linalg.norm(delta) < width * 0.10 or
            np.linalg.norm(np.asarray(scene["origin"]) -
                           (np.asarray(previous_scene["origin"]) + delta)) < width * 0.12 or
            not _score_changed(previous_image, image)):
        return None
    outline = np.asarray(previous_scene["platform"]["outline"], np.float32) + delta
    if cv2.pointPolygonTest(outline.astype(np.float32), tuple(scene["origin"]), True) < width * 0.0125:
        return None

    def materials(frame):
        # Restrict this fallback to the saturated pink frame of the glowing
        # cube. Fill its enclosed pattern holes, never its convex hull/shadow.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        pink = cv2.inRange(hsv, (155, 35, 70), (179, 170, 255))
        contours, _ = cv2.findContours(pink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        masks = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not width ** 2 * 0.025 < area < width ** 2 * 0.12:
                continue
            mask = np.zeros((height, width), np.uint8)
            cv2.drawContours(mask, [contour], -1, 1, -1)
            masks.append(mask)
        return masks

    tx, ty = map(round, previous_scene["target"])
    sources = [m for m in materials(previous_image) if m[ty, tx]]
    if len(sources) != 1:
        return None
    source = sources[0]
    locations = np.round(support).astype(int)
    rim = cv2.dilate(source, np.ones((7, 7), np.uint8))
    local = support[rim[locations[:, 1], locations[:, 0]] > 0]
    if len(local) < 4 or len(np.unique(np.round(local / 4), axis=0)) < 4:
        return None
    matrix = np.float32([[1, 0, delta[0]], [0, 1, delta[1]]])
    moved = cv2.warpAffine(source, matrix, (width, height), flags=cv2.INTER_NEAREST)
    allowed = tracking_mask(image.shape, scene["origin"])
    expected = (moved > 0) & (allowed > 0)
    count = int(np.count_nonzero(expected))
    if count < width ** 2 * 0.025 or count < np.count_nonzero(source) * 0.65:
        return None
    candidates = []
    for mask in materials(image):
        observed = (mask > 0) & (allowed > 0)
        if np.count_nonzero(expected & observed) / max(1, np.count_nonzero(expected | observed)) >= 0.97:
            candidates.append(observed)
    if len(candidates) != 1:
        return None
    interior = cv2.erode((expected & candidates[0]).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    count = int(np.count_nonzero(interior))
    if count < width ** 2 * 0.02:
        return None
    warped = cv2.warpAffine(previous_image, matrix, (width, height))
    gradients = []
    for frame in (warped, image):
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float), (3, 3), 0.8)
        magnitude = np.hypot(cv2.Sobel(gray, cv2.CV_64F, 1, 0), cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        values = magnitude[interior]
        if np.std(values) < 8:
            return None
        gradients.append(values)
    correlation = float(np.corrcoef(*gradients)[0, 1])
    return (correlation, count) if correlation >= 0.80 else None

def _animated_target_alignment(previous_image, previous_scene, image, scene,
                               foreground, delta, points, group):
    """A glowing target may change color; require independent static neighbors."""
    import cv2
    import numpy as np
    height, width = image.shape[:2]
    if not _score_and_player_changed(previous_image, previous_scene, image, scene):
        return None
    _, labels = cv2.connectedComponents(foreground)
    x, y = map(round, previous_scene["target"])
    label = labels[y, x]
    if label == 0:
        return None
    target = (labels == label).astype(np.uint8)
    rim = cv2.dilate(target, np.ones((7, 7), np.uint8))
    locations = np.round(points).astype(int)
    independent = group & (rim[locations[:, 1], locations[:, 0]] == 0)
    support = points[independent]
    if (len(support) < 8 or len(np.unique(np.round(support / 4), axis=0)) < 6 or
            np.min(np.ptp(support, axis=0)) < width * 0.06):
        return None
    matrix = np.float32([[1, 0, delta[0]], [0, 1, delta[1]]])
    moved = cv2.warpAffine(target, matrix, (width, height), flags=cv2.INTER_NEAREST)
    neighborhood = cv2.dilate(moved, np.ones((7, 7), np.uint8)) > 0
    allowed = tracking_mask(image.shape, scene["origin"])
    background = row_background(image)
    current = background_distance(image, background) > 18
    expected = (moved > 0) & (allowed > 0)
    observed = current & neighborhood & (allowed > 0)
    union = expected | observed
    if (np.count_nonzero(expected) < width ** 2 * 0.02 or
            np.count_nonzero(expected & observed) / max(1, np.count_nonzero(union)) < 0.95):
        return None
    visible = cv2.warpAffine(foreground, matrix, (width, height), flags=cv2.INTER_NEAREST)
    visible = (visible > 0) & (allowed > 0)
    # Ignore one antialiased boundary pixel, but keep substantial independent
    # material support instead of accepting a few matching corners or shadows.
    other = cv2.erode((visible & ~neighborhood).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    count = int(np.count_nonzero(other))
    if count < max(width ** 2 * 0.02, np.count_nonzero(visible) * 0.25):
        return None
    warped = cv2.warpAffine(previous_image, matrix, (width, height))
    error = np.max(cv2.absdiff(warped, image), axis=2)
    score = float(np.mean(error[other] < 20))
    return (score, count) if score >= 0.90 else None

def _matching_foreground(previous_image, previous_scene, image):
    """Board appearance used to compare camera hypotheses across sky transitions."""
    import numpy as np
    background = row_background(previous_image)
    foreground = (background_distance(previous_image, background) > 18).astype(np.uint8)
    restrict_tracking_region(foreground, previous_scene["origin"])
    current_background = row_background(image)
    # Use the same maximum-channel color distance as pixel verification.
    # Averaging channels can hide a transition (e.g. one channel changes
    # by 31 while the three-channel average is only 19), leaving recolored
    # shadows to count as platform mismatches.
    background_change = np.max(abs(background - current_background), axis=2)
    # A vertical transition may affect a large band before its whole-frame
    # average crosses the threshold. Require over a quarter of all rows,
    # so a small platform or isolated color noise cannot trigger this.
    if np.mean(background_change) > 20 or np.mean(background_change > 20) > 0.25:
        # Sky transitions recolor cast shadows. Exclude only source pixels
        # consistent with a dimmed background from appearance verification.
        scale = np.sum(previous_image * background, axis=2) / np.maximum(np.sum(background ** 2, axis=2), 1)
        residual = np.max(abs(previous_image - background * scale[:, :, None]), axis=2)
        foreground[(scale > 0.45) & (scale < 0.98) & (residual < 10)] = 0
    return foreground

def _dominant_shift(shifts, tolerance):
    import numpy as np
    return max(shifts, key=lambda delta:
               np.count_nonzero(np.linalg.norm(shifts - delta, axis=1) < tolerance))


def _motion_clusters(points, shifts, tolerance):
    """At most eight independent translation hypotheses, in support order."""
    import numpy as np
    remaining = np.ones(len(shifts), bool)
    for _ in range(8):
        available = shifts[remaining]
        if len(available) < 3:
            break
        center = _dominant_shift(available, tolerance)
        group = remaining & (np.linalg.norm(shifts - center, axis=1) < tolerance)
        remaining[group] = False
        if np.count_nonzero(group) < 3 or len(np.unique(np.round(points[group] / 4), axis=0)) < 3:
            continue
        yield np.median(shifts[group], axis=0), group


def _landing_margin(previous_scene, offset, origin):
    """Signed clearance inside the old target after the estimated camera motion."""
    import cv2
    import numpy as np
    outline = (np.array(previous_scene["platform"]["outline"]) + offset).astype(np.float32)
    return cv2.pointPolygonTest(outline, tuple(origin), True)


def _target_pixel_agreement(previous_image, previous_scene, image, scene, offset):
    """Verify the old target locally when moving shadows spoil the whole scene."""
    import cv2
    import numpy as np
    height, width = image.shape[:2]
    outline = np.asarray(previous_scene["platform"]["outline"], np.int32)
    x0, y0 = outline.min(axis=0)
    x1, y1 = outline.max(axis=0)
    mask = np.zeros((height, width), np.uint8)
    mask[max(0,y0):min(height,y1+round(width*.10)+1),
         max(0,x0):min(width,x1+1)] = 1
    mask &= _matching_foreground(previous_image, previous_scene, image)
    matrix = np.float32([[1,0,offset[0]], [0,1,offset[1]]])
    visible = cv2.warpAffine(mask, matrix, (width,height), flags=cv2.INTER_NEAREST)
    restrict_tracking_region(visible, scene["origin"])
    count = int(np.count_nonzero(visible))
    if count < max(width**2*.005, np.count_nonzero(mask)*.4):
        return 0.0
    warped = cv2.warpAffine(previous_image, matrix, (width,height))
    error = np.max(cv2.absdiff(warped,image), axis=2)
    return float(np.mean(error[visible>0] < 20))


def _check_landing_features(previous_image, previous_scene, image, scene, detailed=False):
    """Track stationary board features to account for camera scrolling."""
    import cv2
    import numpy as np
    if previous_image.shape != image.shape:
        return {"verified": False, "reason": "两次截图尺寸不同"}
    height, width = image.shape[:2]
    extractor = (cv2.SIFT_create(nfeatures=1500, contrastThreshold=0.01, edgeThreshold=20, sigma=0.8)
           if detailed else cv2.ORB_create(nfeatures=1500, fastThreshold=8, edgeThreshold=10))
    features = []
    for frame, state in ((previous_image, previous_scene), (image, scene)):
        # The pawn moves independently of the camera; do not track its texture.
        mask = tracking_mask(frame.shape, state["origin"], value=255)
        features.append(extractor.detectAndCompute(frame, mask))
    (keypoints_a, descriptors_a), (keypoints_b, descriptors_b) = features
    if descriptors_a is None or descriptors_b is None or len(descriptors_b) < 2:
        return {"verified": False, "reason": "无法追踪上一跳平台"}
    pairs = cv2.BFMatcher(cv2.NORM_L2 if detailed else cv2.NORM_HAMMING).knnMatch(descriptors_a, descriptors_b, k=2)
    matches = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.7 * p[1].distance]
    if len(matches) < 6:
        return {"verified": False, "reason": "平台追踪特征不足"}
    points = np.array([keypoints_a[m.queryIdx].pt for m in matches])
    destinations = np.array([keypoints_b[m.trainIdx].pt for m in matches])
    shifts = destinations - points
    tolerance = width * 0.0075
    best = _dominant_shift(shifts, tolerance)
    inliers = np.linalg.norm(shifts - best, axis=1) < tolerance
    unique_points = np.unique(np.round(points[inliers] / 4), axis=0)
    confident = np.count_nonzero(inliers) >= 6 and np.mean(inliers) >= 0.35 and len(unique_points) >= 4
    offset = np.median(shifts[inliers], axis=0)
    margin = _landing_margin(previous_scene, offset, scene["origin"])
    alignment_score = None
    animated_offsets = []
    glowing_offsets = []
    if detailed or not confident or margin < -tolerance:
        # Repeated identical platforms can dominate descriptor matches. Compare
        # competing camera shifts against the visible static scene, without
        # using the pawn's landing location to select a convenient hypothesis.
        foreground = _matching_foreground(previous_image, previous_scene, image)
        original_count = int(np.count_nonzero(foreground))
        hypotheses = []
        for delta, group in _motion_clusters(points, shifts, tolerance):
            matrix = np.float32([[1, 0, delta[0]], [0, 1, delta[1]]])
            warped = cv2.warpAffine(previous_image, matrix, (width, height))
            visible = cv2.warpAffine(foreground, matrix, (width, height), flags=cv2.INTER_NEAREST)
            restrict_tracking_region(visible, scene["origin"])
            count = int(np.count_nonzero(visible))
            small_overlap = count < max(width ** 2 * 0.02, original_count * 0.15)
            # Camera scrolling may remove almost all old platforms. A smaller
            # visible overlap must have more feature and pixel agreement, not
            # merely satisfy a percentage of foreground that has left the frame.
            if count < width ** 2 * 0.01 or (small_overlap and np.count_nonzero(group) < 6):
                continue
            error = np.max(cv2.absdiff(warped, image), axis=2)
            score = float(np.mean(error[visible > 0] < 20))
            # Sparse feature clusters only propose motion: require stronger
            # independent pixel evidence before accepting their displacement.
            minimum_score = 0.95 if small_overlap or np.count_nonzero(group) < 6 else 0.80
            if score < minimum_score and detailed:
                alternate = _animated_target_alignment(previous_image, previous_scene, image, scene,
                                                       foreground, delta, points, group)
                if alternate is None:
                    alternate = _glowing_material_alignment(previous_image, previous_scene, image, scene,
                                                             delta, points, group)
                    if alternate is not None:
                        glowing_offsets.append(delta)
                if alternate is not None:
                    score, count = alternate
                    animated_offsets.append(delta)
                    hypotheses.append((score, delta, group, score * count))
            elif score >= minimum_score:
                hypotheses.append((score, delta, group, score * count))
        # Repeated identical blocks may each match nearly perfectly. Prefer
        # the shift explaining substantially more visible foreground pixels,
        # rather than a tiny overlap with a slightly higher percentage.
        hypotheses.sort(key=lambda item: item[3], reverse=True)
        if (not hypotheses or
                (len(hypotheses) > 1 and hypotheses[0][3] < hypotheses[1][3] * 1.5)):
            # Identical successive blocks can make static endpoint matching
            # ambiguous. A changed score AND displaced pawn supply independent
            # progress evidence. Still require a single high-quality camera
            # hypothesis placing the feet well inside the previous target.
            supported = []
            if _score_and_player_changed(previous_image, previous_scene, image, scene):
                for hypothesis in hypotheses:
                    score, delta, group, explained = hypothesis
                    clearance = _landing_margin(previous_scene, delta, scene["origin"])
                    if (np.count_nonzero(group) >= 6 and
                            delta[1] > tolerance and clearance >= width * 0.0125):
                        # Score progress resolves repeated blocks only with
                        # strong pixel evidence on the actual landing platform.
                        # Keep the scene-wide hypothesis checks and uniqueness.
                        local_score = _target_pixel_agreement(
                            previous_image, previous_scene, image, scene, delta)
                        if score >= 0.95 or local_score >= 0.95:
                            supported.append(hypothesis)
            if len(supported) != 1:
                return {"verified": False, "reason": "平台追踪不一致，场景匹配无法消除歧义"}
            alignment_score, offset, inliers, _ = supported[0]
            margin = _landing_margin(previous_scene, offset, scene["origin"])
            return {"verified": True, "camera_offset": offset.tolist(),
                    "alignment_score": alignment_score, "matched_features": int(np.count_nonzero(inliers)),
                    "landing_margin": margin, "tracking_method": "score-player-and-scene",
                    "reason": "分数、棋子位移及平台匹配共同确认落点"}
        alignment_score, offset, inliers, _ = hypotheses[0]
        margin = _landing_margin(previous_scene, offset, scene["origin"])
    result = {"verified": margin >= -tolerance, "camera_offset": offset.tolist(),
            "alignment_score": alignment_score,
            "matched_features": int(np.count_nonzero(inliers)), "landing_margin": margin,
            "reason": "已确认落在上一目标上" if margin >= -tolerance else "棋子尚未落在上一目标上"}
    if any(np.linalg.norm(offset - candidate) < 0.01 for candidate in animated_offsets):
        result["tracking_method"] = "static-neighbors-and-target-outline"
    if any(np.linalg.norm(offset - candidate) < 0.01 for candidate in glowing_offsets):
        result["tracking_method"] = "glowing-material-outline-and-texture"
    return result

def scene_signature(image, scene=None):
    import cv2
    if scene is not None:
        # A background transition alone must not count as a completed jump.
        player, top = _stability_masks(image, scene)
        image = image.copy()
        image[~(player | top)] = 0
    h = image.shape[0]
    gray = cv2.cvtColor(image[round(h * 0.22):round(h * 0.87)], cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (48, 64), interpolation=cv2.INTER_AREA).reshape(-1).tolist()

def unchanged_scene(previous, current):
    if previous is None:
        return False
    width = current["image_size"][0]
    if any(math.dist(previous[k], current[k]) > width * 0.008 for k in ("origin", "target")):
        return False
    a, b = previous["signature"], current["signature"]
    return len(a) == len(b) and sum(abs(x - y) for x, y in zip(a, b)) / len(a) < 2.0
