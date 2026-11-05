"""Ordered visual recognition API. No GUI and no input injection.

Pixel/geometry measurements feed candidate generators and common validation.
The first nonempty detector result is decisive, including ambiguous results.
Tracking and landing verification have separate modules.
"""

import aiming
from vision_pixels import (
    _color_contours,
    edge_contours,
)
from vision_geometry import (
    _shape,
)
from vision_validation import (
    _exposed_upper_edge,
    _supported_small_surface,
    _platform_candidates,
    _trimmed_upper_edge,
)
from vision_materials import (
    _grid_top_contours,
    _striped_surface,
    _printed_surface,
    _taped_top_contours,
    _shadow_top_contours,
    _concentric_top_contours,
)
from vision_edges import (
    _marked_top_contours,
    _unmarked_textured_top_contours,
)
from vision_small import (
    _shadow_ellipse_contours,
    _two_tone_small_ellipse_contours,
    _shadow_small_diamond_contours,
    _shadow_marked_top_contours,
)
from vision_occlusion import (
    _clipped_diamond_candidates,
    _pawn_occluded_diamond_contours,
    _player_contact_origin,
)
from vision_tracking import (
    _stability_masks,
    scenes_stable,
    _score_and_player_changed,
    _score_changed,
    _glowing_material_alignment,
    _animated_target_alignment,
    _check_landing_features,
    scene_signature,
    unchanged_scene,
)
from vision_landing import (
    check_landing,
    check_landing_recovery,
    _check_contact_landing,
)

class RecognitionError(RuntimeError):
    def __init__(self, message, candidates=None, origin=None):
        super().__init__(message)
        self.candidates = candidates or []
        self.origin = origin

def _candidate_batches(image, origin, angle):
    """Yield validated batches lazily in precedence order.

    A nonempty batch (even an ambiguous one) ends the search. Each generator
    supplies its own evidence; all contours then pass the same validator.
    Contact-based batches explicitly carry their alternative aiming origin.
    """
    import cv2
    width = image.shape[1]

    def validate(contours, detector, *, small=False, contact=None, pawn_origin=None, **metadata):
        candidates = _platform_candidates(
            contours, origin if contact is None else contact, width, angle,
            detector, image, small_surface=small,
            pawn_origin=pawn_origin)
        for candidate in candidates:
            candidate.update(metadata)
        return candidates, contact

    # Observed boundaries and flat regions are preferred to reconstructions.
    contours = edge_contours(image)
    yield validate(contours, "edges")
    for bin_width, clean in ((12, True), (24, True), (12, False), (24, False)):
        yield validate(_color_contours(image, bin_width, clean), "color-regions",
                       color_bin_width=bin_width, color_trim_cleaned=clean)

    # Suppress grain, but validate against the original pixels and boundaries.
    diameter = max(3, round(width * 0.0225)) | 1
    smoothed = cv2.bilateralFilter(image, diameter, 60, width * 0.0375)
    for bin_width in (12, 24):
        yield validate(_color_contours(smoothed, bin_width), "color-regions-smoothed",
                       color_bin_width=bin_width, color_trim_cleaned=True)
    for bin_width in (12, 24):
        yield validate(_shadow_top_contours(image, bin_width, angle), "color-regions-shadow",
                       color_bin_width=bin_width)
    for color_edges, detector in ((False, "marker-and-edges"), (True, "marker-and-color-edges")):
        yield validate(_marked_top_contours(image, origin, angle, color_edges=color_edges), detector)
    for bin_width in (12, 24):
        yield validate(_grid_top_contours(image, bin_width, angle), "white-grid")

    # Small tops require the validator's additional boundary and body evidence.
    compact = list(contours)
    for bin_width in (12, 24):
        compact.extend(_color_contours(image, bin_width, False, min_area_ratio=0.001))
    yield validate(compact, "small-surfaces", small=True)
    yield validate(_shadow_ellipse_contours(image, angle), "shadow-ellipse", small=True)
    yield validate(_shadow_small_diamond_contours(image, angle), "shadow-small-diamond", small=True)

    # Clipping and occlusion have distinct observed-edge requirements.
    yield _clipped_diamond_candidates(image, origin, angle), None
    # Coarser color bins can reconnect antialiased edges of the same material.
    for bin_width in (12, 24):
        yield validate(_taped_top_contours(image, angle, bin_width), "material-seam")
    yield validate(_concentric_top_contours(image, angle), "concentric-rings")
    for color_edges in (False, True):
        yield validate(_marked_top_contours(image, origin, angle, color_edges=color_edges,
                                           partial_edge=True), "marker-and-three-edges")
    yield _clipped_diamond_candidates(image, origin, angle, allow_trim=True), None
    yield validate(_shadow_marked_top_contours(image, angle, "ellipse"), "shadow-marked-ellipse")
    yield validate(_pawn_occluded_diamond_contours(image, origin, angle), "pawn-occluded-diamond",
                   pawn_origin=origin)

    for bin_width in (12, 24):
        yield validate(_color_contours(image, bin_width, True, min_area_ratio=0.001),
                       "small-surfaces-cleaned", small=True)
    yield validate(_two_tone_small_ellipse_contours(image, angle), "two-tone-small-ellipse", small=True)
    yield validate(_shadow_marked_top_contours(image, angle, "ellipse", tiny_marker=True),
                   "tiny-shadow-marked-ellipse", small=True)

    contact = _player_contact_origin(image, origin)
    if contact is not None:
        tiny = list(contours)
        for bin_width in (12, 24):
            for clean in (False, True):
                tiny.extend(_color_contours(image, bin_width, clean, min_area_ratio=0.001))
        tiny = [c for c in tiny if cv2.boundingRect(c)[2] <= width * 0.12]
        yield validate(tiny, "small-surfaces-contact", small=True, contact=contact)

        # Reuse the marker/edge reconstruction at small scale. The fitted foot
        # center avoids inflating short jumps with the pawn's lowest pixel.
        for color_edges in (False, True):
            yield validate(_marked_top_contours(image, origin, angle, color_edges=color_edges,
                                               partial_edge=True, small_surface=True),
                           "small-marker-and-three-edges", small=True, contact=contact)

    yield validate(_shadow_ellipse_contours(image, angle, small_surface=False), "shadow-ellipse")
    yield validate(_unmarked_textured_top_contours(image, origin, angle), "four-observed-edges")
    yield validate(_marked_top_contours(image, origin, angle, printed_texture=True),
                   "marker-and-printed-surface")


def detect_scene(image, angle=30):
    import numpy as np
    height, width = image.shape[:2]
    origin = aiming.find_player(image)
    if origin is None:
        raise RecognitionError("未能可靠找到棋子，可能尚未开始、仍在动画中或已经结束。")
    candidates, aim_origin = [], None
    for candidates, aim_origin in _candidate_batches(image, origin, angle):
        if candidates:
            break
    if len(candidates) != 1:
        raise RecognitionError(f"找到 {len(candidates)} 个可靠目标，不能唯一确定下一平台。已停止，未盲跳。",
                               candidates=candidates, origin=origin)
    selected = candidates[0]
    # The lowest purple pixel lies below the contact plane and inflates
    # projected travel. Revalidate close shadow ovals using the fitted foot
    # center, retaining the calibrated origin for other detector paths.
    if (selected["detector"] == "shadow-ellipse" and
            selected["projection"]["horizontal"] < width * 0.15):
        contact = _player_contact_origin(image, origin)
        if contact is None:
            raise RecognitionError("近距离阴影圆台的脚底接触点不明确，未发送跳跃。",
                                   candidates=candidates, origin=origin)
        checked = _platform_candidates(
            [np.asarray(selected["outline"], np.int32).reshape(-1, 1, 2)],
            contact, width, angle, "shadow-ellipse", image,
            small_surface=np.ptp(np.asarray(selected["outline"])[:, 0]) + 1 <= width * 0.16)
        if len(checked) != 1:
            raise RecognitionError("近距离阴影圆台的接触点投影未通过校验，未发送跳跃。",
                                   candidates=candidates, origin=origin)
        selected = checked[0]
        aim_origin = contact
    scene = {"origin": origin, "target": selected["center"], "projection": selected["projection"],
             "platform": selected, "image_size": [width, height]}
    if aim_origin is not None:
        scene["aim_origin"] = aim_origin
    return scene

def annotate(image, scene, angle):
    import cv2
    import numpy as np
    marked, _ = aiming.draw_guides(image, scene.get("aim_origin",scene["origin"]), scene["target"], angle)
    outline = np.array(scene["platform"]["outline"], dtype=np.int32)
    cv2.polylines(marked, [outline], True, (0, 220, 0), 2)
    return marked
