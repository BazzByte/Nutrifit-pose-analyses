# =============================================================================
# core/pose_detector.py
# غلاف لـ MediaPipe Pose — سهل الاستخدام وسريع
# Lightweight MediaPipe Pose wrapper
#
# لماذا MediaPipe وليس YOLOv8?
# Why MediaPipe over YOLOv8?
# ─────────────────────────────────────────────────────────────────────────────
# ✓ أسرع على CPU (30+ FPS على أجهزة متوسطة مقابل 10-15 لـ YOLOv8-pose)
# ✓ لا يحتاج GPU أو تثبيت CUDA
# ✓ يُعطي إحداثيات 3D (عمق تقريبي) — مفيد لتقدير الزوايا الجانبية
# ✓ يُعطي 33 landmark دقيقة (كافية لكلا التمرينين)
# ✓ مدمج في مكتبة واحدة خفيفة (mediapipe) — سهل النشر على Flutter backend
# ✗ YOLOv8 أدق مع تجمعات بشرية أو أوضاع غير تقليدية — لكن هذا غير مطلوب هنا
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


# فهارس Landmarks في MediaPipe (33 نقطة)
# MediaPipe Pose landmark indices
class MP_LANDMARKS:
    NOSE          = 0
    LEFT_EYE      = 2
    RIGHT_EYE     = 5
    LEFT_EAR      = 7
    RIGHT_EAR     = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER= 12
    LEFT_ELBOW    = 13
    RIGHT_ELBOW   = 14
    LEFT_WRIST    = 15
    RIGHT_WRIST   = 16
    LEFT_HIP      = 23
    RIGHT_HIP     = 24
    LEFT_KNEE     = 25
    RIGHT_KNEE    = 26
    LEFT_ANKLE    = 27
    RIGHT_ANKLE   = 28
    LEFT_HEEL     = 29
    RIGHT_HEEL    = 30
    LEFT_FOOT     = 31
    RIGHT_FOOT    = 32


@dataclass
class PoseResult:
    """نتيجة معالجة frame واحد / Result from processing a single frame."""
    detected:   bool                    # هل تم اكتشاف شخص؟
    landmarks:  Optional[object]        # MediaPipe landmark list
    world_lm:   Optional[object]        # World landmarks (3D متري تقريبي)
    frame_w:    int
    frame_h:    int
    confidence: float                   # نسبة الثقة العامة 0..1


class PoseDetector:
    """
    كلاس خفيف يُغلّف MediaPipe Pose.
    Lightweight wrapper around mediapipe.solutions.pose.Pose.
    """

    def __init__(self,
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.6,
                 min_tracking_confidence: float = 0.5,
                 smooth_landmarks: bool = True) -> None:
        """
        Args:
            model_complexity:            0 = أسرع, 1 = متوازن, 2 = أدق (أبطأ)
                                         0=fastest, 1=balanced, 2=accurate(slow)
            min_detection_confidence:    حد أدنى للكشف الأولي
            min_tracking_confidence:     حد أدنى للتتبع المستمر
            smooth_landmarks:            تنعيم اهتزاز Landmarks
        """
        self._mp_pose = mp.solutions.pose
        self._pose    = self._mp_pose.Pose(
            model_complexity          = model_complexity,
            min_detection_confidence  = min_detection_confidence,
            min_tracking_confidence   = min_tracking_confidence,
            smooth_landmarks          = smooth_landmarks,
            enable_segmentation       = False,   # لا نحتاجها — توفير وقت
        )

    # ------------------------------------------------------------------ #
    def process(self, frame: np.ndarray) -> PoseResult:
        """
        اعالج frame واحد وأرجع PoseResult.
        Process a single BGR frame and return a PoseResult.

        Args:
            frame: صورة BGR من OpenCV

        Returns:
            PoseResult مع الـ landmarks أو detected=False
        """
        h, w = frame.shape[:2]

        # MediaPipe يحتاج RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._pose.process(rgb)

        if results.pose_landmarks is None:
            return PoseResult(
                detected=False, landmarks=None, world_lm=None,
                frame_w=w, frame_h=h, confidence=0.0
            )

        # احسب متوسط visibility كمؤشر ثقة
        visibilities = [lm.visibility for lm in results.pose_landmarks.landmark]
        avg_conf = float(np.mean(visibilities))

        return PoseResult(
            detected   = True,
            landmarks  = results.pose_landmarks.landmark,
            world_lm   = (results.pose_world_landmarks.landmark
                          if results.pose_world_landmarks else None),
            frame_w    = w,
            frame_h    = h,
            confidence = avg_conf,
        )

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """أغلق الموارد / Release MediaPipe resources."""
        self._pose.close()

    # ------------------------------------------------------------------ #
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
