# =============================================================================
# core/angle_utils.py
# دوال حساب الزوايا والمسافات
# Geometry helpers for angle / distance calculations
# =============================================================================

import math
from typing import Tuple, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# النوع الأساسي: نقطة ثنائية الأبعاد أو ثلاثية
# Basic point type (2-D or 3-D from MediaPipe)
# --------------------------------------------------------------------------- #
Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]


def calculate_angle(a: Point2D | Point3D,
                    b: Point2D | Point3D,
                    c: Point2D | Point3D) -> float:
    """
    احسب الزاوية عند النقطة b بين الشعاعين (b→a) و (b→c).
    Compute the angle at vertex b between rays b→a and b→c.

    المعادلة: angle = arccos( (BA · BC) / (|BA| * |BC|) )
    يُرجع: الزاوية بالدرجات (0..180)
    Returns angle in degrees [0, 180].
    """
    a_arr = np.array(a[:2], dtype=float)
    b_arr = np.array(b[:2], dtype=float)
    c_arr = np.array(c[:2], dtype=float)

    ba = a_arr - b_arr
    bc = c_arr - b_arr

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def calculate_angle_3d(a: Point3D, b: Point3D, c: Point3D) -> float:
    """
    نفس calculate_angle لكن يستخدم الإحداثيات الثلاثية من MediaPipe.
    Same as calculate_angle but uses full 3-D coordinates from MediaPipe.
    أدق في تقدير الزوايا من الجانب.
    More accurate for side-view angle estimation.
    """
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    c_arr = np.array(c, dtype=float)

    ba = a_arr - b_arr
    bc = c_arr - b_arr

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0

    cosine = float(np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def midpoint(a: Point2D, b: Point2D) -> Point2D:
    """نقطة المنتصف بين نقطتين / Midpoint between two 2-D points."""
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def euclidean_distance(a: Point2D, b: Point2D) -> float:
    """المسافة الإقليدية بين نقطتين / Euclidean distance between two 2-D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def normalize_angle(angle: float) -> float:
    """
    يُعيد الزاوية إلى المجال [0, 180].
    Clamp angle to [0, 180].
    """
    return float(np.clip(angle, 0.0, 180.0))


def angle_to_score(angle: float,
                   ideal_min: float,
                   ideal_max: float,
                   tolerance: float = 15.0) -> float:
    """
    حوّل زاوية إلى نسبة مئوية (0..1) بناءً على النطاق المثالي.
    Convert an angle to a 0..1 score based on the ideal range.

    - إذا كانت الزاوية داخل [ideal_min, ideal_max] → 1.0 (ممتاز)
    - إذا كانت خارج النطاق بمقدار ≤ tolerance → انخفاض خطي
    - If within ideal range → 1.0 (perfect)
    - If outside range by ≤ tolerance → linear decay to 0
    """
    if ideal_min <= angle <= ideal_max:
        return 1.0

    deviation = max(ideal_min - angle, angle - ideal_max)
    score = max(0.0, 1.0 - deviation / tolerance)
    return float(score)


def smooth_value(history: list[float], new_val: float, window: int = 5) -> float:
    """
    متوسط متحرك بسيط لتنعيم القيم وتقليل الـ noise.
    Simple moving average to smooth noisy angle readings.
    يُحدّث القائمة في المكان ويُرجع المتوسط.
    Updates history in-place and returns the smoothed value.
    """
    history.append(new_val)
    if len(history) > window:
        history.pop(0)
    return float(np.mean(history))


def landmark_to_pixel(landmark,
                      frame_width: int,
                      frame_height: int) -> Tuple[int, int]:
    """
    حوّل Landmark من MediaPipe (قيم نسبية 0..1) إلى إحداثيات البكسل.
    Convert a MediaPipe landmark (normalized 0..1) to pixel coordinates.
    """
    return (int(landmark.x * frame_width),
            int(landmark.y * frame_height))


def landmarks_to_pixels(landmarks,
                        frame_width: int,
                        frame_height: int,
                        indices: Sequence[int]) -> list[Tuple[int, int]]:
    """
    حوّل قائمة من الـ landmarks إلى قائمة إحداثيات البكسل دفعة واحدة.
    Batch-convert a list of landmark indices to pixel coords.
    """
    return [landmark_to_pixel(landmarks[i], frame_width, frame_height)
            for i in indices]


def get_3d_point(landmark) -> Point3D:
    """استخرج إحداثيات 3D من MediaPipe landmark."""
    return (landmark.x, landmark.y, landmark.z)
