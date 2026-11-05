"""Landing verification and conservative recovery from tracking ambiguity."""

import math
from vision_occlusion import (
    _player_contact_origin,
)
from vision_tracking import (
    _check_landing_features,
    _score_and_player_changed,
    _score_changed,
)

def check_landing(previous_image, previous_scene, image, scene):
    """Use detailed local features only when the normal tracker cannot verify."""
    result = _check_landing_features(previous_image, previous_scene, image, scene)
    if result["verified"]:
        return result
    detailed = _check_landing_features(previous_image, previous_scene, image, scene, detailed=True)
    if detailed["verified"]:
        detailed.setdefault("tracking_method", "detailed-features-and-pixels")
        return detailed
    # A contact-origin jump must be checked against the same physical foot
    # point. The lowest rendered pixel can lie below a tiny top even when the
    # foot center is inside it. Keep this separate from camera estimation:
    # both trackers must independently agree, with strong pixel evidence.
    contact_landing = _check_contact_landing(
        previous_image, previous_scene, image, scene, result, detailed)
    if contact_landing is not None:
        return contact_landing
    return result

def check_landing_recovery(previous_image, previous_scene, image, scene):
    """Independent body correspondence and camera-relative jump verification.

    The worker must also require a stable multi-frame observation. Search the
    whole playing field before considering the feet: do not pick a convenient
    camera shift merely because it puts the pawn inside the old target.
    """
    import cv2
    import numpy as np
    def refuse(reason):
        return {"verified": False, "reason": reason, "tracking_method": "body-and-world-motion"}
    if previous_image.shape != image.shape:
        return refuse("重新核对截图尺寸不同")
    if not _score_changed(previous_image, image):
        return refuse("尚无明确的分数变化")
    height, width = image.shape[:2]
    outline = np.asarray(previous_scene["platform"]["outline"], np.float32)
    low, high = outline.min(axis=0), outline.max(axis=0)
    # Keep lateral boundaries and a strip below the top. The new pawn hides
    # the top/center marker, but should never hide the platform's side wall.
    pad = max(2, round(width * 0.01))
    full_x0, full_x1 = int(low[0]) - pad, int(high[0]) + pad + 1
    x0, x1 = max(0, full_x0), min(width, full_x1)
    y0 = int(high[1]) + max(2, round(width * 0.0075))
    y1 = y0 + round(width * 0.10)
    # An edge-clipped target still has useful visible side walls. Match only
    # observed pixels; never synthesize the missing part of its appearance.
    if (y0 < height * 0.3 or y1 >= height * 0.87 or x1 - x0 < width * 0.05 or
            x1 - x0 < (full_x1 - full_x0) * 0.6):
        return refuse("上一目标可见侧面不足")
    ox, oy = previous_scene["origin"]
    if x0 - width * 0.06 < ox < x1 + width * 0.06 and oy > y0 and oy - width * 0.25 < y1:
        return refuse("上一目标侧面被棋子遮挡")
    template = previous_image[y0:y1, x0:x1]
    # Uniform white patches cannot establish a unique correspondence.
    if float(np.std(template.astype(float), axis=(0, 1)).max()) < 12:
        return refuse("上一目标侧面缺少可区分的外观")
    field_top, field_bottom = round(height * 0.3), round(height * 0.87)
    field = image[field_top:field_bottom]
    if field.shape[0] < template.shape[0]:
        return refuse("当前可搜索画面不足")
    match, reason = _match_platform_body(previous_image, image, (x0, y0, x1, y1),
                                         field_top, field_bottom)
    if match is None:
        return refuse(reason)
    mx, my = match.pop("location")
    offset = np.asarray([mx-x0, my-y0], np.float32)
    moved = outline + offset
    margin = cv2.pointPolygonTest(moved, tuple(scene["origin"]), True)
    contact_evidence = {}
    if (margin < width * 0.006 and np.ptp(outline[:, 0]) <= width * 0.16 and
            match.get("alignment_score", 0) >= 0.95):
        # On a tiny top the pawn's lowest pixel sits below its contact plane.
        # The body match is already unique; recover only a measured foot arc,
        # retaining score, world-motion and worker stability requirements.
        contact = _player_contact_origin(image, scene["origin"])
        if contact is not None:
            contact_margin = cv2.pointPolygonTest(moved, contact, True)
            if contact_margin >= width * 0.006:
                contact_evidence = {"lowest_pixel_margin": margin, "landing_contact": contact}
                margin = contact_margin
    if margin < width * 0.006 or np.linalg.norm(offset) < width * 0.02:
        return refuse("棋子未位于匹配台面内部或平台未移动")
    # Camera recentering may leave the pawn at the same screen coordinates.
    # Measure its displacement relative to the matched stationary platform,
    # then check forward travel along the configured isometric jump direction.
    old_origin = np.asarray(previous_scene.get("aim_origin", previous_scene["origin"]), np.float32)
    travel = np.asarray(scene.get("aim_origin", scene["origin"]), np.float32) - old_origin - offset
    projection = previous_scene.get("projection", {})
    angle = math.radians(projection.get("angle_degrees", 30))
    direction = 1 if previous_scene["target"][0] > old_origin[0] else -1
    axis = np.asarray([direction * math.cos(angle), -math.sin(angle)])
    along = float(travel @ axis)
    cross = abs(float(travel[0] * axis[1] - travel[1] * axis[0]))
    if along < width * 0.05 or cross > max(width * 0.025, along * 0.10):
        return refuse("平台位移与本次跳跃方向不一致")
    return {"verified": True, "camera_offset": offset.tolist(), "landing_margin": margin,
            "world_travel": travel.tolist(), "direction_error": cross,
            "template_visible_fraction": (x1-x0)/(full_x1-full_x0),
            **match, **contact_evidence,
            "reason": "分数变化、唯一平台侧面和相对平台的棋子位移确认落点"}

def _match_platform_body(previous, current, bounds, field_top, field_bottom):
    """Find a unique side wall without using the pawn to choose the match.

    Prefer unchanged pixels. Lighting/activation can change a material's color;
    in that case require both a unique gradient match and interior texture.
    A silhouette alone is deliberately insufficient for repeated plain blocks.
    """
    import cv2
    import numpy as np
    from vision_pixels import row_background, background_distance

    x0, y0, x1, y1 = bounds
    template = previous[y0:y1, x0:x1]
    field = current[field_top:field_bottom]
    radius = max(4, round(current.shape[1] * 0.0125))

    def peak(scores, minimum=False):
        scores = np.where(np.isfinite(scores), scores, 1 if minimum else -1)
        low, high, low_xy, high_xy = cv2.minMaxLoc(scores)
        value, (x, y) = (low, low_xy) if minimum else (high, high_xy)
        scores[max(0,y-radius):y+radius+1, max(0,x-radius):x+radius+1] = 1 if minimum else -1
        return value, float(scores.min() if minimum else scores.max()), (x, y)

    best, second, (mx, my) = peak(cv2.matchTemplate(field, template, cv2.TM_SQDIFF_NORMED), True)
    if best <= 0.003:
        if second < max(best * 2.5, best + 0.002):
            return None, "多个平台侧面相似，无法唯一匹配"
        matched = field[my:my+template.shape[0], mx:mx+template.shape[1]]
        agreement = float(np.mean(np.max(cv2.absdiff(template, matched), axis=2) < 20))
        if agreement >= 0.95:
            return dict(location=(mx,my+field_top), alignment_score=agreement,
                        template_error=best, alternative_error=second,
                        tracking_method="unique-body-and-world-motion"), None

    def gradients(image):
        gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32), (3,3), .8)
        return cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0), cv2.Sobel(gray,cv2.CV_32F,0,1))

    old_gradient, new_gradient = gradients(previous), gradients(current)
    reference = old_gradient[y0:y1,x0:x1]
    best, second, (mx,my) = peak(cv2.matchTemplate(
        new_gradient[field_top:field_bottom], reference, cv2.TM_CCOEFF_NORMED))
    if best < 0.90 or best-second < 0.15:
        return None, "平台侧面结构不足或存在相似匹配"
    my += field_top
    # Exclude the silhouette and crop border before measuring print/grain.
    material = (background_distance(previous, row_background(previous)) > 30).astype(np.uint8)
    interior = cv2.erode(material, np.ones((9,9),np.uint8))[y0:y1,x0:x1].astype(bool)
    interior[:4] = interior[-4:] = False
    interior[:,:4] = interior[:,-4:] = False
    observed = new_gradient[my:my+reference.shape[0], mx:mx+reference.shape[1]]
    if np.count_nonzero(interior) < max(100, reference.size * .30):
        return None, "平台侧面内部纹理可见不足"
    a, b = reference[interior], observed[interior]
    if min(float(a.std()), float(b.std())) < 8:
        return None, "平台侧面内部纹理不足"
    correlation = float(np.corrcoef(a,b)[0,1])
    if not np.isfinite(correlation) or correlation < 0.80:
        return None, "平台侧面内部纹理不一致"
    return dict(location=(mx,my), alignment_score=best, texture_correlation=correlation,
                alternative_score=second, tracking_method="unique-body-texture-and-world-motion"), None


def _check_contact_landing(previous_image, previous_scene, image, scene, normal, detailed):
    import cv2
    import numpy as np
    if "aim_origin" not in previous_scene or previous_image.shape != image.shape:
        return None
    width = image.shape[1]
    outline = np.asarray(previous_scene["platform"]["outline"], np.float32)
    if np.ptp(outline[:, 0]) > width * 0.16:
        return None
    for result in (normal, detailed):
        if (result.get("alignment_score") is None or result["alignment_score"] < 0.95 or
                result.get("matched_features", 0) < 6 or "camera_offset" not in result):
            return None
    if math.dist(normal["camera_offset"], detailed["camera_offset"]) > width * 0.005:
        return None
    if not _score_and_player_changed(previous_image, previous_scene, image, scene):
        return None
    old_contact = _player_contact_origin(previous_image, previous_scene["origin"])
    contact = _player_contact_origin(image, scene["origin"])
    if (old_contact is None or contact is None or
            math.dist(old_contact, previous_scene["aim_origin"]) > width * 0.0025):
        return None
    margins = [cv2.pointPolygonTest(
        (outline + np.asarray(r["camera_offset"], np.float32)), contact, True)
        for r in (normal, detailed)]
    if min(margins) < width * 0.006:
        return None
    return {**normal, "verified": True, "lowest_pixel_margin": normal["landing_margin"],
            "landing_margin": min(margins), "landing_contact": contact,
            "tracking_method": "contact-center-and-two-trackers",
            "reason": "分数变化及双重平台匹配确认脚底中心已落在小台面内"}
