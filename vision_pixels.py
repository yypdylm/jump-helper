"""Shared pixel measurements; thresholds belong to the caller's evidence checks.

Imports stay lazy so command-line help and input diagnostics need no OpenCV.
"""


def row_background(image):
    """Estimate the sky independently per row, preserving vertical gradients."""
    import numpy as np
    return np.median(image, axis=1)[:, None, :]


def background_distance(image, background):
    """Maximum channel distance, not a mean that can hide a changed channel."""
    import numpy as np
    return np.max(abs(image.astype(float) - background), axis=2)


def color_codes(image, bin_width):
    """Pack quantized BGR channels into the same uint16 region labels."""
    import numpy as np
    quantized = image.astype(np.uint16) // bin_width
    radix = (256 + bin_width - 1) // bin_width
    return quantized[:, :, 0] * radix * radix + quantized[:, :, 1] * radix + quantized[:, :, 2]


def material_projection(pixels, base, norm=None):
    """Brightness along a pigment's color ray and maximum channel residual.

    Accepts an Nx3 sample or an HxWx3 image. Callers retain their own minimum
    pigment norm, brightness range and residual/coverage requirements.
    """
    import numpy as np
    if norm is None:
        norm = np.dot(base, base)
    brightness = pixels @ base / norm
    residual = np.max(abs(pixels - brightness[..., None] * base), axis=-1)
    return brightness, residual


def blend_projection(pixels, base, delta, norm):
    """Position and residual on a fixed color-to-color line (e.g. white ink)."""
    import numpy as np
    blend = (pixels - base) @ delta / norm
    error = np.max(abs(pixels - (base + blend[..., None] * delta)), axis=-1)
    return blend, error


def paint_pawn_region(mask, origin, value):
    """Paint the common bounded pawn region in-place; clip at image edges."""
    import cv2
    width = mask.shape[1]
    x, y = origin
    cv2.rectangle(mask, (round(x - width * 0.06), round(y - width * 0.25)),
                  (round(x + width * 0.06), round(y + width * 0.025)), value, -1)
    return mask


def bounded_pawn_mask(image, origin):
    import cv2
    import numpy as np
    pawn = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV),
                       (110, 30, 30), (170, 255, 160))
    allowed = paint_pawn_region(np.zeros(image.shape[:2], np.uint8), origin, 255)
    return pawn & allowed


def restrict_tracking_region(mask, origin):
    """Keep static board pixels and exclude the independently moving pawn."""
    height = mask.shape[0]
    mask[:round(height * 0.3)] = 0
    mask[round(height * 0.87):] = 0
    return paint_pawn_region(mask, origin, 0)


def tracking_mask(shape, origin, value=1):
    import numpy as np
    return restrict_tracking_region(np.full(shape[:2], value, np.uint8), origin)



def _color_contours(image, bin_width=12, clean_trim=True, min_area_ratio=0.0025):
    """Flat top surfaces can survive where connected decoration breaks Canny edges."""
    import cv2
    import numpy as np
    height, width = image.shape[:2]
    top, bottom = round(height * 0.22), round(height * 0.87)
    codes = color_codes(image[top:bottom], bin_width)
    size = max(3, round(width * 0.0125)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    contours = []
    for code, count in zip(*np.unique(codes, return_counts=True)):
        if count < min_area_ratio * width ** 2:
            continue
        mask = (codes == code).astype(np.uint8) * 255
        # Remove thin trim attached to a platform top without filling missing faces.
        if clean_trim:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(0, top))
        contours.extend(found)
    return contours


def channel_edges(image):
    """Boundaries visible in any color channel, including equal-luminance colors."""
    import cv2
    import numpy as np
    edges = np.zeros(image.shape[:2], np.uint8)
    for channel in cv2.split(cv2.GaussianBlur(image, (3, 3), 0)):
        edges |= cv2.Canny(channel, 25, 60)
    return edges


def edge_contours(image):
    """Primary board contours; deliberately retain internal boundary contours."""
    import cv2
    import numpy as np
    height = image.shape[0]
    edges = channel_edges(image)
    edges[:round(0.22 * height)] = 0
    edges[round(0.87 * height):] = 0
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours
