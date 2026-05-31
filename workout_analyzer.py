# =============================================================================
# workout_analyzer.py  — v2
# الكلاس الرئيسي للتحليل — يدعم 5 تمارين + ثنائي اللغة
# Main WorkoutAnalyzer — 5 exercises + bilingual (AR/EN)
#
# الاستخدام / Usage:
#   analyzer = WorkoutAnalyzer(lang="ar")
#   result   = analyzer.process_frame(frame, "pushup")
#
# الدالة البسيطة المتوافقة مع المواصفات:
#   frame_out, pct, feedback, angles = analyzer.process_frame_simple(frame, "squat")
# =============================================================================

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np

from core.pose_detector import PoseDetector
from exercises.base_exercise import ExerciseResult
from exercises.pushup_analyzer import PushupAnalyzer
from exercises.pullup_analyzer import PullupAnalyzer
from exercises.jumpingjack_analyzer import JumpingJackAnalyzer
from exercises.crunch_analyzer import CrunchAnalyzer
from exercises.squat_analyzer import SquatAnalyzer
from visualization.skeleton_drawer import HUDRenderer
from visualization.i18n import i18n as _global_i18n, I18n, Lang


# أنواع التمارين المقبولة
ExerciseType = Literal["pushup", "pullup", "jumpingjack", "crunch", "squat"]
ALL_EXERCISES = ("pushup", "pullup", "jumpingjack", "crunch", "squat")


@dataclass
class AnalysisResult:
    """
    النتيجة الكاملة من process_frame — جاهزة للإرسال إلى Flutter.
    Full result from process_frame — Flutter-friendly.
    """
    frame:             np.ndarray
    percentage:        float
    feedback_text:     str
    audio_cue:         Optional[str]
    angles:            Dict[str, float]
    rep_count:         int
    phase:             str
    is_calibrating:    bool
    feedback_history:  list
    fps:               float
    exercise_type:     str
    language:          str

    def to_dict(self) -> dict:
        """
        للإرسال عبر HTTP أو Method Channel (بدون الـ frame).
        For HTTP / Method Channel transmission (frame excluded).
        """
        return {
            "percentage":       self.percentage,
            "feedback_text":    self.feedback_text,
            "audio_cue":        self.audio_cue,
            "angles":           self.angles,
            "rep_count":        self.rep_count,
            "phase":            self.phase,
            "is_calibrating":   self.is_calibrating,
            "feedback_history": self.feedback_history,
            "fps":              round(self.fps, 1),
            "exercise_type":    self.exercise_type,
            "language":         self.language,
        }


# =============================================================================
class WorkoutAnalyzer:
    """
    المحلل الرئيسي — يدعم 5 تمارين وثنائي اللغة.
    Main workout analyzer — 5 exercises, bilingual.

    التمارين المدعومة / Supported exercises:
    - pushup      → تمرين الضغط
    - pullup      → تمرين العقلة
    - jumpingjack → القفز المتفرج
    - crunch      → تمرين البطن
    - squat       → تمرين الجلسة
    """

    def __init__(self,
                 model_complexity: int = 1,
                 lang: Lang = "ar",
                 show_fps: bool = True) -> None:
        """
        Args:
            model_complexity: 0=سريع, 1=متوازن (افتراضي), 2=دقيق
            lang:             "ar" (عربي) أو "en" (إنجليزي)
            show_fps:         عرض FPS على الشاشة
        """
        self._lang      = lang
        self._show_fps  = show_fps
        self._detector  = PoseDetector(model_complexity=model_complexity)
        self._i18n      = I18n(lang=lang)

        # إنشاء محلل لكل تمرين
        self._analyzers: Dict[str, object] = {
            "pushup":       PushupAnalyzer(lang=lang),
            "pullup":       PullupAnalyzer(lang=lang),
            "jumpingjack":  JumpingJackAnalyzer(lang=lang),
            "crunch":       CrunchAnalyzer(lang=lang),
            "squat":        SquatAnalyzer(lang=lang),
        }

        self._hud = HUDRenderer(i18n=self._i18n)

        # قياس FPS
        self._fps_buf:  list[float] = []
        self._last_ts:  float       = time.time()

    # ------------------------------------------------------------------ #
    def process_frame(self,
                      frame: np.ndarray,
                      exercise_type: ExerciseType) -> AnalysisResult:
        """
        الدالة الرئيسية — معالجة frame واحد.
        Primary method — process a single camera frame.

        Args:
            frame:          BGR frame from OpenCV
            exercise_type:  أحد: pushup | pullup | jumpingjack | crunch | squat

        Returns:
            AnalysisResult كامل
        """
        if exercise_type not in self._analyzers:
            raise ValueError(
                f"exercise_type غير صالح: '{exercise_type}'. "
                f"الخيارات: {ALL_EXERCISES}"
            )

        fps         = self._compute_fps()
        pose_result = self._detector.process(frame)
        analyzer    = self._analyzers[exercise_type]
        ex_result: ExerciseResult = analyzer.analyze(frame, pose_result)

        final_frame = self._hud.render(
            ex_result,
            exercise_type = exercise_type,
            fps           = fps if self._show_fps else 0.0,
        )

        return AnalysisResult(
            frame            = final_frame,
            percentage       = ex_result.percentage,
            feedback_text    = ex_result.feedback_text,
            audio_cue        = ex_result.audio_cue,
            angles           = ex_result.angles,
            rep_count        = ex_result.rep_count,
            phase            = ex_result.phase_label,
            is_calibrating   = ex_result.is_calibrating,
            feedback_history = ex_result.feedback_history,
            fps              = fps,
            exercise_type    = exercise_type,
            language         = self._lang,
        )

    # ------------------------------------------------------------------ #
    def process_frame_simple(
        self,
        frame: np.ndarray,
        exercise_type: ExerciseType,
    ) -> Tuple[np.ndarray, float, str, Dict[str, float]]:
        """
        واجهة بسيطة متوافقة مع المواصفات الأصلية.
        Simple interface matching the original spec.

        Returns:
            (annotated_frame, percentage, feedback_text, angles_dict)
        """
        r = self.process_frame(frame, exercise_type)
        return r.frame, r.percentage, r.feedback_text, r.angles

    # ------------------------------------------------------------------ #
    def set_language(self, lang: Lang) -> None:
        """
        غيّر اللغة في أثناء التشغيل لجميع المحللين.
        Switch language at runtime for all analyzers.
        """
        self._lang = lang
        self._i18n.set_lang(lang)
        self._hud.i18n.set_lang(lang)
        for analyzer in self._analyzers.values():
            analyzer.set_language(lang)

    # ------------------------------------------------------------------ #
    def reset_exercise(self, exercise_type: ExerciseType) -> None:
        """إعادة تعيين تمرين محدد / Reset a specific exercise."""
        if exercise_type in self._analyzers:
            self._analyzers[exercise_type].reset()

    def reset_all(self) -> None:
        """إعادة تعيين كل التمارين / Reset all exercises."""
        for a in self._analyzers.values():
            a.reset()

    def get_rep_count(self, exercise_type: ExerciseType) -> int:
        """استرجع عدد التكرارات / Get rep count for an exercise."""
        a = self._analyzers.get(exercise_type)
        if a and a._rep_counter:
            return a._rep_counter.rep_count
        return 0

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """أطلق الموارد / Release resources."""
        self._detector.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()

    # ------------------------------------------------------------------ #
    def _compute_fps(self) -> float:
        now  = time.time()
        dt   = now - self._last_ts
        self._last_ts = now
        if dt > 0:
            self._fps_buf.append(1.0 / dt)
        if len(self._fps_buf) > 30:
            self._fps_buf.pop(0)
        return float(np.mean(self._fps_buf)) if self._fps_buf else 0.0
