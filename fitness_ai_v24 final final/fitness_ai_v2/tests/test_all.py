# =============================================================================
# tests/test_all.py
# اختبارات الوحدات الشاملة — تستخدم fixtures من conftest.py
# Comprehensive unit tests — uses conftest.py fixtures
#
# تشغيل / Run:
#   pytest tests/ -v
#   pytest tests/ -v --cov=. --cov-report=term-missing
#   pytest tests/test_all.py::TestI18n -v          ← وحدة محددة
# =============================================================================

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# conftest.py يضيف ROOT لـ sys.path تلقائياً
# conftest.py auto-adds ROOT to sys.path


# =============================================================================
# ── core/angle_utils ─────────────────────────────────────────────────────────
# =============================================================================
from core.angle_utils import (
    calculate_angle, angle_to_score, smooth_value,
    euclidean_distance, midpoint, normalize_angle, landmark_to_pixel,
)


class TestAngleUtils:
    """اختبارات دوال الزوايا / Angle utility tests."""

    def test_right_angle(self):
        angle = calculate_angle((0, 1), (0, 0), (1, 0))
        assert abs(angle - 90.0) < 0.5, f"Expected ~90, got {angle}"

    def test_straight_line_180(self):
        angle = calculate_angle((0, 0), (1, 0), (2, 0))
        assert abs(angle - 180.0) < 0.5

    def test_zero_angle(self):
        angle = calculate_angle((1, 0), (0, 0), (1, 0))
        assert angle == 0.0

    def test_symmetry(self):
        a1 = calculate_angle((1, 0), (0, 0), (0, 1))
        a2 = calculate_angle((0, 1), (0, 0), (1, 0))
        assert abs(a1 - a2) < 0.001

    def test_score_perfect(self):
        s = angle_to_score(90.0, 80.0, 100.0)
        assert s == 1.0

    def test_score_outside_tolerance(self):
        s = angle_to_score(50.0, 80.0, 100.0, tolerance=30.0)
        assert 0.0 <= s < 1.0

    def test_score_far_zero(self):
        s = angle_to_score(0.0, 80.0, 100.0, tolerance=30.0)
        assert s == 0.0

    def test_smooth_reduces_noise(self):
        history = []
        vals = [10, 100, 10, 100, 10, 100]
        results = [smooth_value(history, v, window=6) for v in vals]
        assert (max(results) - min(results)) < 90

    def test_euclidean_3_4_5(self):
        assert abs(euclidean_distance((0, 0), (3, 4)) - 5.0) < 0.001

    def test_midpoint(self):
        assert midpoint((0, 0), (4, 4)) == (2.0, 2.0)

    def test_normalize_clamps(self):
        assert normalize_angle(-5)  == 0.0
        assert normalize_angle(200) == 180.0
        assert normalize_angle(90)  == 90.0

    def test_landmark_to_pixel(self):
        lm = MagicMock()
        lm.x, lm.y = 0.5, 0.5
        assert landmark_to_pixel(lm, 640, 480) == (320, 240)

    def test_landmark_to_pixel_corner(self):
        lm = MagicMock()
        lm.x, lm.y = 0.0, 0.0
        assert landmark_to_pixel(lm, 640, 480) == (0, 0)


# =============================================================================
# ── core/rep_counter ─────────────────────────────────────────────────────────
# =============================================================================
from core.rep_counter import RepCounter, RepPhase


class TestRepCounter:
    """اختبارات عداد التكرارات / Rep counter tests."""

    def test_initial_state(self):
        rc = RepCounter()
        assert rc.rep_count == 0
        assert rc.phase == RepPhase.IDLE

    def test_no_rep_at_top_only(self):
        rc = RepCounter(top_threshold=155.0, bottom_threshold=100.0)
        for _ in range(10):
            rc.update(165.0)
        assert rc.rep_count == 0

    def test_no_rep_at_bottom_only(self):
        rc = RepCounter(top_threshold=155.0, bottom_threshold=100.0)
        for _ in range(10):
            rc.update(80.0)
        assert rc.rep_count == 0

    def test_full_rep_completes(self):
        rc = RepCounter(top_threshold=155.0, bottom_threshold=100.0)
        rc.cfg.confirmation_frames = 1
        rc.cfg.min_rep_interval    = 0.0
        # أعلى → أسفل → أعلى
        for _ in range(3): rc.update(165.0)
        for _ in range(3): rc.update(85.0)
        completed = any(rc.update(165.0) for _ in range(5))
        assert completed
        assert rc.rep_count == 1

    def test_reset_clears_everything(self):
        rc = RepCounter()
        rc.rep_count = 7
        rc._reached_bottom = True
        rc.reset()
        assert rc.rep_count == 0
        assert not rc._reached_bottom

    def test_phase_label_is_string(self):
        rc = RepCounter()
        assert isinstance(rc.phase_label, str)
        assert len(rc.phase_label) > 0


# =============================================================================
# ── visualization/i18n ───────────────────────────────────────────────────────
# =============================================================================
class TestI18n:
    """اختبارات الترجمة / i18n tests."""

    def test_arabic_contains_arabic_chars(self, i18n_ar):
        text = i18n_ar.t("calibrating")
        assert any('\u0600' <= c <= '\u06FF' for c in text), \
            f"Expected Arabic chars in '{text}'"

    def test_english_is_ascii(self, i18n_en):
        text = i18n_en.t("calibrating")
        assert text and text.isascii(), f"Expected ASCII text, got '{text}'"

    def test_ar_en_differ(self, i18n_ar, i18n_en):
        assert i18n_ar.t("reps") != i18n_en.t("reps")

    def test_switch_at_runtime(self, i18n_ar):
        ar_text = i18n_ar.t("reps")
        i18n_ar.set_lang("en")
        en_text = i18n_ar.t("reps")
        assert ar_text != en_text
        i18n_ar.set_lang("ar")   # restore

    def test_missing_key_returns_key(self, i18n_ar):
        result = i18n_ar.t("this_key_does_not_exist_xyz")
        assert result == "this_key_does_not_exist_xyz"

    def test_is_arabic_flag(self, i18n_ar, i18n_en):
        assert i18n_ar.is_arabic
        assert not i18n_en.is_arabic

    @pytest.mark.parametrize("key", [
        "pu_sagging", "pu_piking", "pu_excellent", "pu_good",
        "plu_swinging", "plu_excellent", "plu_go_higher",
        "jj_arms_up", "jj_legs_wide", "jj_excellent",
        "cr_chin_chest", "cr_go_higher", "cr_excellent",
        "sq_knees_out", "sq_back_straight", "sq_excellent",
        "calibrating", "hold_start_pos", "no_person", "reps", "form_score",
    ])
    def test_all_keys_exist(self, key, i18n_ar):
        result = i18n_ar.t(key)
        assert result != key, f"Key '{key}' not found in i18n dictionary"

    def test_all_keys_have_english(self, key="calibrating"):
        from visualization.i18n import I18n
        i = I18n("en")
        assert i.t(key) != key


# =============================================================================
# ── config/thresholds ────────────────────────────────────────────────────────
# =============================================================================
from config.thresholds import (
    PUSHUP_CFG, PULLUP_CFG, JUMPJACK_CFG,
    CRUNCH_CFG, SQUAT_CFG, REP_CFG, VIZ_CFG,
)


class TestThresholds:
    """اختبارات الإعدادات / Config tests."""

    @pytest.mark.parametrize("cfg,name", [
        (PUSHUP_CFG,   "pushup"),
        (PULLUP_CFG,   "pullup"),
        (JUMPJACK_CFG, "jumpingjack"),
        (CRUNCH_CFG,   "crunch"),
        (SQUAT_CFG,    "squat"),
    ])
    def test_weights_sum_100(self, cfg, name):
        total = sum(cfg.weights.values())
        assert abs(total - 100.0) < 0.001, \
            f"{name} weights sum = {total}, expected 100"

    def test_pushup_elbow_range_valid(self):
        assert PUSHUP_CFG.elbow_down_min < PUSHUP_CFG.elbow_down_max
        assert PUSHUP_CFG.elbow_up_min   < PUSHUP_CFG.elbow_up_max
        assert PUSHUP_CFG.elbow_down_max < PUSHUP_CFG.elbow_up_min

    def test_pullup_elbow_range_valid(self):
        assert PULLUP_CFG.elbow_top_min    < PULLUP_CFG.elbow_top_max
        assert PULLUP_CFG.elbow_bottom_min < PULLUP_CFG.elbow_bottom_max

    def test_rep_counter_config_positive(self):
        assert REP_CFG.confirmation_frames >= 1
        assert REP_CFG.min_rep_interval    >= 0.0
        assert REP_CFG.min_angle_delta     >  0.0


# =============================================================================
# ── exercises: analyze() with fixtures ───────────────────────────────────────
# =============================================================================
class TestPushupAnalyzer:
    def test_no_detection_returns_zero(self, blank_frame, mock_pose_not_detected):
        from exercises.pushup_analyzer import PushupAnalyzer
        a = PushupAnalyzer(lang="en")
        r = a.analyze(blank_frame, mock_pose_not_detected)
        assert r.percentage == 0.0
        assert isinstance(r.feedback_text, str)

    def test_detected_returns_valid(self, blank_frame, mock_pose_detected):
        from exercises.pushup_analyzer import PushupAnalyzer
        a = PushupAnalyzer(lang="ar")
        r = a.analyze(blank_frame, mock_pose_detected)
        assert 0.0 <= r.percentage <= 100.0
        assert r.annotated_frame.shape == blank_frame.shape
        assert isinstance(r.rep_count, int)
        assert isinstance(r.angles, dict)
        assert "elbow" in r.angles

    def test_reset_clears_reps(self):
        from exercises.pushup_analyzer import PushupAnalyzer
        a = PushupAnalyzer()
        a._rep_counter.rep_count = 10
        a.reset()
        assert a._rep_counter.rep_count == 0

    def test_arabic_feedback(self, blank_frame, mock_pose_not_detected):
        from exercises.pushup_analyzer import PushupAnalyzer
        a = PushupAnalyzer(lang="ar")
        r = a.analyze(blank_frame, mock_pose_not_detected)
        assert isinstance(r.feedback_text, str)
        assert len(r.feedback_text) > 0


class TestPullupAnalyzer:
    def test_no_detection(self, blank_frame, mock_pose_not_detected):
        from exercises.pullup_analyzer import PullupAnalyzer
        a = PullupAnalyzer(lang="en")
        r = a.analyze(blank_frame, mock_pose_not_detected)
        assert r.percentage == 0.0

    def test_detected_valid(self, blank_frame, mock_pose_detected):
        from exercises.pullup_analyzer import PullupAnalyzer
        a = PullupAnalyzer(lang="ar")
        r = a.analyze(blank_frame, mock_pose_detected)
        assert 0.0 <= r.percentage <= 100.0
        assert "elbow" in r.angles


class TestJumpingJackAnalyzer:
    def test_no_detection(self, blank_frame, mock_pose_not_detected):
        from exercises.jumpingjack_analyzer import JumpingJackAnalyzer
        a = JumpingJackAnalyzer(lang="ar")
        r = a.analyze(blank_frame, mock_pose_not_detected)
        assert r.percentage == 0.0

    def test_detected_has_leg_ratio(self, blank_frame, mock_pose_detected):
        from exercises.jumpingjack_analyzer import JumpingJackAnalyzer
        a = JumpingJackAnalyzer(lang="ar")
        r = a.analyze(blank_frame, mock_pose_detected)
        assert "leg_ratio" in r.angles
        assert 0.0 <= r.percentage <= 100.0


class TestCrunchAnalyzer:
    def test_no_detection(self, blank_frame, mock_pose_not_detected):
        from exercises.crunch_analyzer import CrunchAnalyzer
        a = CrunchAnalyzer(lang="ar")
        r = a.analyze(blank_frame, mock_pose_not_detected)
        assert r.percentage == 0.0

    def test_detected_has_hip_angle(self, blank_frame, mock_pose_detected):
        from exercises.crunch_analyzer import CrunchAnalyzer
        a = CrunchAnalyzer(lang="en")
        r = a.analyze(blank_frame, mock_pose_detected)
        assert "hip" in r.angles
        assert 0.0 <= r.percentage <= 100.0


class TestSquatAnalyzer:
    def test_no_detection(self, blank_frame, mock_pose_not_detected):
        from exercises.squat_analyzer import SquatAnalyzer
        a = SquatAnalyzer(lang="ar")
        r = a.analyze(blank_frame, mock_pose_not_detected)
        assert r.percentage == 0.0

    def test_detected_has_knee_angle(self, blank_frame, mock_pose_detected):
        from exercises.squat_analyzer import SquatAnalyzer
        a = SquatAnalyzer(lang="ar")
        r = a.analyze(blank_frame, mock_pose_detected)
        assert "knee" in r.angles

    def test_valgus_ratio_in_angles(self, blank_frame, mock_pose_detected):
        from exercises.squat_analyzer import SquatAnalyzer
        a = SquatAnalyzer()
        r = a.analyze(blank_frame, mock_pose_detected)
        assert "valgus_ratio" in r.angles


# =============================================================================
# ── WorkoutAnalyzer integration ──────────────────────────────────────────────
# =============================================================================
class TestWorkoutAnalyzer:
    """اختبارات تكامل / Integration tests."""

    @patch("core.pose_detector.PoseDetector.process")
    def test_all_exercises_arabic(self, mock_proc, blank_frame, mock_pose_not_detected):
        """كل التمارين تعمل بالعربي / All exercises work in Arabic."""
        from workout_analyzer import WorkoutAnalyzer, ALL_EXERCISES
        mock_proc.return_value = mock_pose_not_detected

        with WorkoutAnalyzer(model_complexity=0, lang="ar") as az:
            for ex in ALL_EXERCISES:
                r = az.process_frame(blank_frame, ex)
                assert r is not None
                assert r.language == "ar"
                assert isinstance(r.feedback_text, str)

    @patch("core.pose_detector.PoseDetector.process")
    def test_all_exercises_english(self, mock_proc, blank_frame, mock_pose_not_detected):
        """كل التمارين تعمل بالإنجليزي / All exercises work in English."""
        from workout_analyzer import WorkoutAnalyzer, ALL_EXERCISES
        mock_proc.return_value = mock_pose_not_detected

        with WorkoutAnalyzer(model_complexity=0, lang="en") as az:
            for ex in ALL_EXERCISES:
                r = az.process_frame(blank_frame, ex)
                assert r.language == "en"

    @patch("core.pose_detector.PoseDetector.process")
    def test_language_switch_runtime(self, mock_proc, blank_frame, mock_pose_not_detected):
        """تبديل اللغة في أثناء التشغيل / Runtime language switch."""
        from workout_analyzer import WorkoutAnalyzer
        mock_proc.return_value = mock_pose_not_detected

        with WorkoutAnalyzer(lang="ar") as az:
            r_ar = az.process_frame(blank_frame, "pushup")
            az.set_language("en")
            r_en = az.process_frame(blank_frame, "pushup")
            assert r_ar.language == "ar"
            assert r_en.language == "en"

    @patch("core.pose_detector.PoseDetector.process")
    def test_invalid_exercise_raises(self, mock_proc, blank_frame, mock_pose_not_detected):
        """تمرين غير صالح يرفع ValueError / Invalid exercise raises ValueError."""
        from workout_analyzer import WorkoutAnalyzer
        mock_proc.return_value = mock_pose_not_detected

        with WorkoutAnalyzer() as az:
            with pytest.raises(ValueError, match="غير صالح"):
                az.process_frame(blank_frame, "invalid_xyz")  # type: ignore

    @patch("core.pose_detector.PoseDetector.process")
    def test_to_dict_json_serializable(self, mock_proc, blank_frame, mock_pose_not_detected):
        """النتيجة قابلة لـ JSON / Result dict is JSON-serializable."""
        from workout_analyzer import WorkoutAnalyzer
        mock_proc.return_value = mock_pose_not_detected

        with WorkoutAnalyzer(lang="en") as az:
            r = az.process_frame(blank_frame, "squat")
            d = r.to_dict()
            s = json.dumps(d)
            assert "percentage" in d
            assert "feedback_text" in d
            assert "rep_count" in d

    @patch("core.pose_detector.PoseDetector.process")
    def test_reset_exercise(self, mock_proc, blank_frame, mock_pose_not_detected):
        """إعادة التعيين تعمل / Reset works correctly."""
        from workout_analyzer import WorkoutAnalyzer
        mock_proc.return_value = mock_pose_not_detected

        with WorkoutAnalyzer() as az:
            az._analyzers["pushup"]._rep_counter.rep_count = 15
            az.reset_exercise("pushup")
            assert az.get_rep_count("pushup") == 0

    @patch("core.pose_detector.PoseDetector.process")
    def test_simple_interface_returns_tuple(self, mock_proc, blank_frame, mock_pose_not_detected):
        """الواجهة البسيطة ترجع tuple / Simple interface returns 4-tuple."""
        from workout_analyzer import WorkoutAnalyzer
        mock_proc.return_value = mock_pose_not_detected

        with WorkoutAnalyzer() as az:
            result = az.process_frame_simple(blank_frame, "crunch")
            assert isinstance(result, tuple)
            assert len(result) == 4
            frame_out, pct, feedback, angles = result
            assert isinstance(frame_out, np.ndarray)
            assert isinstance(pct, float)
            assert isinstance(feedback, str)
            assert isinstance(angles, dict)


# =============================================================================
# ── HUD Renderer ─────────────────────────────────────────────────────────────
# =============================================================================
class TestHUDRenderer:
    """اختبارات واجهة HUD / HUD renderer tests."""

    def _make_result(self, blank_frame, is_calib=False, pct=75.0):
        from exercises.base_exercise import ExerciseResult
        return ExerciseResult(
            annotated_frame  = blank_frame.copy(),
            percentage       = pct,
            feedback_text    = "اختبار / Test",
            audio_cue        = None,
            angles           = {"elbow": 90.0, "knee": 120.0},
            rep_count        = 5,
            phase_label      = "TOP",
            is_calibrating   = is_calib,
            feedback_history = ["msg1", "msg2", "msg3"],
        )

    def test_render_returns_same_shape(self, blank_frame):
        from visualization.skeleton_drawer import HUDRenderer
        hud = HUDRenderer()
        r   = self._make_result(blank_frame)
        out = hud.render(r, "pushup", fps=30.0)
        assert out.shape == blank_frame.shape

    def test_calibration_overlay(self, blank_frame):
        from visualization.skeleton_drawer import HUDRenderer
        hud = HUDRenderer()
        r   = self._make_result(blank_frame, is_calib=True, pct=0.0)
        out = hud.render(r, "squat")
        assert out.shape == blank_frame.shape   # لا ينهار / no crash

    @pytest.mark.parametrize("exercise", ["pushup","pullup","jumpingjack","crunch","squat"])
    def test_all_exercise_types(self, blank_frame, exercise):
        from visualization.skeleton_drawer import HUDRenderer
        hud = HUDRenderer()
        r   = self._make_result(blank_frame, pct=60.0)
        out = hud.render(r, exercise, fps=25.0)
        assert out is not None
        assert out.shape == blank_frame.shape

    @pytest.mark.parametrize("lang", ["ar", "en"])
    def test_both_languages(self, blank_frame, lang):
        from visualization.skeleton_drawer import HUDRenderer
        from visualization.i18n import I18n
        hud = HUDRenderer(i18n=I18n(lang))
        r   = self._make_result(blank_frame)
        out = hud.render(r, "pushup")
        assert out.shape == blank_frame.shape


# =============================================================================
# ── arabic_renderer ──────────────────────────────────────────────────────────
# =============================================================================
class TestArabicRenderer:
    """اختبارات محرك النصوص العربية / Arabic renderer tests."""

    def test_renderer_status_is_string(self):
        from visualization.arabic_renderer import renderer
        assert isinstance(renderer.status, str)
        assert len(renderer.status) > 0

    def test_put_text_arabic_no_crash(self, blank_frame):
        from visualization.arabic_renderer import renderer
        out = renderer.put_text(blank_frame.copy(), "مرحبا بك", (10, 50),
                                font_size=24, color=(255, 255, 255))
        assert out is not None
        assert out.shape == blank_frame.shape

    def test_put_text_english_no_crash(self, blank_frame):
        from visualization.arabic_renderer import renderer
        out = renderer.put_text(blank_frame.copy(), "Hello World", (10, 50),
                                font_size=24, color=(0, 255, 0))
        assert out.shape == blank_frame.shape

    def test_put_text_centered_no_crash(self, blank_frame):
        from visualization.arabic_renderer import renderer
        out = renderer.put_text_centered(blank_frame.copy(), "وسط الشاشة",
                                         y=100, font_size=22)
        assert out.shape == blank_frame.shape

    def test_get_text_size_returns_positive(self):
        from visualization.arabic_renderer import renderer
        w, h = renderer.get_text_size("اختبار", 22)
        assert w > 0
        assert h > 0

    def test_empty_text_returns_frame(self, blank_frame):
        from visualization.arabic_renderer import renderer
        out = renderer.put_text(blank_frame.copy(), "", (10, 50))
        assert out.shape == blank_frame.shape

    def test_arabic_supported_property(self):
        from visualization.arabic_renderer import renderer
        assert isinstance(renderer.arabic_supported, bool)
