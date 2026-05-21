# =============================================================================
# tests/conftest.py
# إعدادات مشتركة لجميع الاختبارات (pytest fixtures)
# Shared fixtures for all tests
# =============================================================================

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# أضف مجلد المشروع الجذر لمسار البحث حتى تعمل الـ imports
# Add project root to sys.path so all imports work correctly
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures مشتركة / Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def blank_frame():
    """إطار فيديو فارغ 640×480 / Blank 640×480 BGR frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def blank_frame_hd():
    """إطار 1280×720 / HD blank frame."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def mock_pose_detected():
    """
    PoseResult وهمي مع landmarks / Mock PoseResult with 33 landmarks.
    النقاط موزعة بشكل معقول لشخص واقف.
    Points distributed like a standing person.
    """
    from core.pose_detector import PoseResult

    lm = []
    # توزيع نقاط تقريبي لشخص واقف أمام الكاميرا
    # Approximate landmark positions for a person standing facing camera
    positions = {
        # رأس / Head
        0:  (0.50, 0.08),   # nose
        2:  (0.48, 0.07),   # left eye
        5:  (0.52, 0.07),   # right eye
        7:  (0.46, 0.09),   # left ear
        8:  (0.54, 0.09),   # right ear
        # كتفان / Shoulders
        11: (0.42, 0.25),   # left shoulder
        12: (0.58, 0.25),   # right shoulder
        # كوعان / Elbows
        13: (0.38, 0.40),   # left elbow
        14: (0.62, 0.40),   # right elbow
        # رسغان / Wrists
        15: (0.36, 0.55),   # left wrist
        16: (0.64, 0.55),   # right wrist
        # وركان / Hips
        23: (0.44, 0.55),   # left hip
        24: (0.56, 0.55),   # right hip
        # ركبتان / Knees
        25: (0.44, 0.72),   # left knee
        26: (0.56, 0.72),   # right knee
        # كاحلان / Ankles
        27: (0.44, 0.88),   # left ankle
        28: (0.56, 0.88),   # right ankle
        # قدمان / Feet
        29: (0.43, 0.92),   # left heel
        30: (0.57, 0.92),   # right heel
        31: (0.43, 0.95),   # left foot
        32: (0.57, 0.95),   # right foot
    }

    for i in range(33):
        m = MagicMock()
        x, y = positions.get(i, (0.50 + (i % 5) * 0.01, 0.50 + (i % 7) * 0.02))
        m.x = x
        m.y = y
        m.z = 0.0
        m.visibility = 0.95
        lm.append(m)

    return PoseResult(
        detected   = True,
        landmarks  = lm,
        world_lm   = None,
        frame_w    = 640,
        frame_h    = 480,
        confidence = 0.95,
    )


@pytest.fixture
def mock_pose_not_detected():
    """PoseResult بدون كشف / PoseResult with no detection."""
    from core.pose_detector import PoseResult
    return PoseResult(False, None, None, 640, 480, 0.0)


@pytest.fixture
def i18n_ar():
    """I18n عربي / Arabic I18n instance."""
    from visualization.i18n import I18n
    return I18n("ar")


@pytest.fixture
def i18n_en():
    """I18n إنجليزي / English I18n instance."""
    from visualization.i18n import I18n
    return I18n("en")
