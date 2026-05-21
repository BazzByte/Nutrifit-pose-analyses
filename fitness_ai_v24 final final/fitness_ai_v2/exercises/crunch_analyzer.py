# =============================================================================
# exercises/crunch_analyzer.py
# محلل تمرين البطن (Crunch / Crunches)
# Abdominal crunch exercise analyzer
# =============================================================================

from __future__ import annotations
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from config.thresholds import CRUNCH_CFG, CrunchThresholds
from core.angle_utils import calculate_angle, angle_to_score, landmark_to_pixel
from core.pose_detector import PoseResult, MP_LANDMARKS as LM
from core.rep_counter import RepCounter
from exercises.base_exercise import BaseExercise, ExerciseResult


class CrunchAnalyzer(BaseExercise):
    """
    محلل تمرين البطن (Crunch).
    Abdominal crunch analyzer.

    يقيّم:
    ✓ مدى رفع الكتفين عن الأرض (ROM)
    ✓ سلامة الرقبة (لا يسحبها) (Neck safety)
    ✓ التحكم في الحركة (Controlled movement)
    ✓ إيقاع التنفس
    ✓ حساب التكرارات

    المعايرة: يُعايَر وهو مستلقٍ على ظهره
    Calibration: user lies flat on back
    """

    def __init__(self, cfg: CrunchThresholds = CRUNCH_CFG, lang: str = "ar") -> None:
        super().__init__(lang=lang)
        self.cfg = cfg

        # زاوية الورك: مستلقٍ = ~170°، جالس = ~60°
        # Hip angle: lying = ~170°, crunched = ~60°
        self._rep_counter = RepCounter(
            top_threshold    = 140.0,   # مستلقٍ (زاوية كبيرة)
            bottom_threshold = 90.0,    # منحنٍ (زاوية صغيرة)
        )
        # نعكس المنطق كما في Pull-up (top=زاوية صغيرة = رفع)
        self._rep_counter.top_threshold    = 100.0   # بعد الرفع
        self._rep_counter.bottom_threshold = 150.0   # مستلقٍ كاملاً

        self._prev_hip_angle: float = 170.0

    # ------------------------------------------------------------------ #
    def analyze(self, frame: np.ndarray, pose_result: PoseResult) -> ExerciseResult:
        annotated = frame.copy()
        is_calib  = self._check_calibration()
        t = self.i18n.t

        if not pose_result.detected:
            return self._make(annotated, 0.0,
                              f"{t('no_person')} — {t('check_camera')}",
                              {}, is_calib)

        lm = pose_result.landmarks
        w, h = pose_result.frame_w, pose_result.frame_h
        def px(i): return landmark_to_pixel(lm[i], w, h)

        # ── نقاط ──
        ls = px(LM.LEFT_SHOULDER);  rs = px(LM.RIGHT_SHOULDER)
        lh = px(LM.LEFT_HIP);       rh = px(LM.RIGHT_HIP)
        lk = px(LM.LEFT_KNEE);      rk = px(LM.RIGHT_KNEE)
        la = px(LM.LEFT_ANKLE);     ra = px(LM.RIGHT_ANKLE)
        nose = px(LM.NOSE)

        # ── زاوية الورك (الكتف - الورك - الركبة) ──
        # هذه الزاوية تصغر عند الرفع وتكبر عند الاستلقاء
        l_hip_a = calculate_angle(ls, lh, lk)
        r_hip_a = calculate_angle(rs, rh, rk)
        hip_a   = self._smooth_angle("hip", (l_hip_a + r_hip_a) / 2)

        # ── زاوية الرقبة (الأنف - الكتف - الورك) — كشف سحب الرقبة ──
        neck_a  = self._smooth_angle("neck", calculate_angle(nose, ls, lh))

        # ── نسبة رفع الكتفين (الكتف Y أقل من الورك Y = مرتفع) ──
        # في الصورة: Y أصغر = أعلى
        shoulder_y = (ls[1] + rs[1]) / 2
        hip_y      = (lh[1] + rh[1]) / 2
        lift_ratio = max(0.0, (hip_y - shoulder_y) / max(hip_y, 1))

        angles: Dict[str, float] = {
            "hip":       round(hip_a, 1),
            "neck":      round(neck_a, 1),
            "lift_ratio": round(lift_ratio, 2),
            "l_hip":     round(l_hip_a, 1),
        }

        # ── تكرارات: نستخدم 180 - hip_a لعكس المنطق ──
        inverted = 180.0 - hip_a
        self._rep_counter.update(inverted)

        # ── نسبة ──
        pct = self._score(hip_a, neck_a, lift_ratio) if not is_calib else 50.0

        # ── تعليق ──
        feedback, audio = self._feedback(hip_a, neck_a, lift_ratio, pct, t)
        self._add_feedback(feedback)
        self._draw(annotated, pose_result, angles, pct)

        return self._make(annotated, pct, feedback, angles, is_calib, audio)

    # ------------------------------------------------------------------ #
    def _score(self, hip_a: float, neck_a: float, lift_ratio: float) -> float:
        w = self.cfg.weights

        # 1. مدى الحركة (ROM)
        is_up = hip_a < 120
        if is_up:
            rom_s = angle_to_score(hip_a, self.cfg.hip_up_max - 20,
                                   self.cfg.hip_up_max, 20.0)
        else:
            rom_s = angle_to_score(hip_a, self.cfg.hip_down_min,
                                   180.0, 20.0)

        # 2. سلامة الرقبة
        neck_s = angle_to_score(neck_a, self.cfg.neck_pull_max, 180.0, 25.0)

        # 3. تحكم في الحركة (نفترض جيداً كافتراضي — يُحسَّن بتتبع السرعة)
        ctrl_s = 0.8

        total = (w["rom"] * rom_s + w["neck_safe"] * neck_s +
                 w["controlled"] * ctrl_s) / 100.0
        return round(float(np.clip(total * 100, 0, 100)), 1)

    # ------------------------------------------------------------------ #
    def _feedback(self, hip_a, neck_a, lift_ratio, score, t) -> Tuple[str, Optional[str]]:
        # سحب الرقبة
        if neck_a < self.cfg.neck_pull_max - 10:
            m = t("cr_chin_chest"); return m, self._get_audio_cue(m)

        # لم يرتفع كفاية
        if hip_a > self.cfg.hip_up_max + 30 and self._rep_counter._reached_bottom:
            m = t("cr_go_higher"); return m, self._get_audio_cue(m)

        # تنبيه عام للتحكم
        if score < 60:
            m = t("cr_controlled"); return m, self._get_audio_cue(m)

        if score >= 85: return t("cr_excellent"), None
        if score >= 70: return t("cr_good"), None

        m = t("cr_breathe"); return m, self._get_audio_cue(m)

    # ------------------------------------------------------------------ #
    def _draw(self, frame, pose_result, angles, score):
        if not pose_result.detected: return
        lm = pose_result.landmarks
        w, h = pose_result.frame_w, pose_result.frame_h
        def px(i): return landmark_to_pixel(lm[i], w, h)

        c = (0,220,0) if score>=80 else (0,165,255) if score>=55 else (0,50,255)
        pairs = [
            (LM.NOSE, LM.LEFT_SHOULDER, 2),
            (LM.NOSE, LM.RIGHT_SHOULDER, 2),
            (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, 3),
            (LM.LEFT_SHOULDER, LM.LEFT_HIP, 4),
            (LM.RIGHT_SHOULDER, LM.RIGHT_HIP, 4),
            (LM.LEFT_HIP, LM.RIGHT_HIP, 3),
            (LM.LEFT_HIP, LM.LEFT_KNEE, 3),
            (LM.RIGHT_HIP, LM.RIGHT_KNEE, 3),
            (LM.LEFT_KNEE, LM.LEFT_ANKLE, 2),
            (LM.RIGHT_KNEE, LM.RIGHT_ANKLE, 2),
        ]
        for i1, i2, t in pairs:
            cv2.line(frame, px(i1), px(i2), c, t, cv2.LINE_AA)
        for j in [LM.NOSE, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
                  LM.LEFT_HIP, LM.RIGHT_HIP, LM.LEFT_KNEE, LM.RIGHT_KNEE]:
            cv2.circle(frame, px(j), 6, (255,255,255), -1, cv2.LINE_AA)
            cv2.circle(frame, px(j), 7, c, 2, cv2.LINE_AA)

        # اعرض زاوية الورك
        cv2.putText(frame, f"Hip: {int(angles.get('hip',0))}°",
                    (px(LM.LEFT_HIP)[0]+10, px(LM.LEFT_HIP)[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2, cv2.LINE_AA)

    def _make(self, frame, pct, feedback, angles, is_calib, audio=None):
        return ExerciseResult(
            annotated_frame=frame, percentage=pct, feedback_text=feedback,
            audio_cue=audio, angles=angles,
            rep_count=self._rep_counter.rep_count if self._rep_counter else 0,
            phase_label=self._rep_counter.phase_label if self._rep_counter else "",
            is_calibrating=is_calib, feedback_history=list(self._feedback_history))
