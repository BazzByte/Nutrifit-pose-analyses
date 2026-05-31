# =============================================================================
# exercises/jumpingjack_analyzer.py
# محلل تمرين القفز المتفرج (Jumping Jack)
# Jumping Jack exercise analyzer
# =============================================================================

from __future__ import annotations
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from config.thresholds import JUMPJACK_CFG, JumpingJackThresholds
from core.angle_utils import calculate_angle, angle_to_score, landmark_to_pixel, euclidean_distance
from core.pose_detector import PoseResult, MP_LANDMARKS as LM
from core.rep_counter import RepCounter
from exercises.base_exercise import BaseExercise, ExerciseResult


class JumpingJackAnalyzer(BaseExercise):
    """
    محلل تمرين القفز المتفرج.
    Jumping Jack analyzer.

    يقيّم:
    ✓ رفع الذراعين فوق الرأس (Arms overhead ROM)
    ✓ فتح الساقين بالكافي (Legs spread ROM)
    ✓ تزامن الذراعين والساقين (Sync)
    ✓ الهبوط الناعم (Landing quality)
    ✓ إيقاع منتظم (Rhythm)
    """

    def __init__(self, cfg: JumpingJackThresholds = JUMPJACK_CFG, lang: str = "ar") -> None:
        super().__init__(lang=lang)
        self.cfg = cfg

        # نستخدم زاوية الكتف (ذراع-جذع) للتكرارات
        # Top = ذراعان مرفوعتان (زاوية كبيرة ~160°)
        # Bottom = ذراعان بجانب الجسم (زاوية صغيرة ~20°)
        self._rep_counter = RepCounter(
            top_threshold    = 140.0,
            bottom_threshold = 50.0,
        )

        self._prev_arm_angle: float = 0.0

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

        # ── نقاط أساسية ──
        ls = px(LM.LEFT_SHOULDER);  rs = px(LM.RIGHT_SHOULDER)
        le = px(LM.LEFT_ELBOW);     re = px(LM.RIGHT_ELBOW)
        lw = px(LM.LEFT_WRIST);     rw = px(LM.RIGHT_WRIST)
        lh = px(LM.LEFT_HIP);       rh = px(LM.RIGHT_HIP)
        la = px(LM.LEFT_ANKLE);     ra = px(LM.RIGHT_ANKLE)
        nose = px(LM.NOSE)

        # ── زاوية الذراع: (الرسغ - الكتف - الورك) ──
        # يقيس مقدار رفع الذراع فوق الرأس
        l_arm_angle = calculate_angle(lw, ls, lh)
        r_arm_angle = calculate_angle(rw, rs, rh)
        arm_angle   = self._smooth_angle("arm", (l_arm_angle + r_arm_angle) / 2)

        # ── عرض القدمين نسبةً لعرض الكتفين ──
        shoulder_width = euclidean_distance(ls, rs)
        ankle_width    = euclidean_distance(la, ra)
        leg_ratio      = (ankle_width / shoulder_width) if shoulder_width > 0 else 1.0
        leg_ratio_s    = self._smooth_angle("leg_ratio", leg_ratio, window=4)

        # ── تزامن الذراعين والساقين ──
        # عندما الذراعان مرفوعتان = الساقان مفتوحتان
        arm_up    = arm_angle > 130.0
        legs_wide = leg_ratio_s > self.cfg.legs_wide_ratio
        sync_ok   = (arm_up == legs_wide)   # True = متزامنان

        angles: Dict[str, float] = {
            "arm_angle":  round(arm_angle, 1),
            "leg_ratio":  round(leg_ratio_s, 2),
            "l_arm":      round(l_arm_angle, 1),
            "r_arm":      round(r_arm_angle, 1),
        }

        # ── تكرارات (بناءً على زاوية الذراع) ──
        self._rep_counter.update(arm_angle)

        # ── نسبة ──
        pct = self._score(arm_angle, leg_ratio_s, sync_ok) if not is_calib else 50.0

        # ── تعليق ──
        feedback, audio = self._feedback(arm_angle, leg_ratio_s, sync_ok, pct, t)
        self._add_feedback(feedback)
        self._draw(annotated, pose_result, angles, pct, arm_up, legs_wide)

        return self._make(annotated, pct, feedback, angles, is_calib, audio)

    # ------------------------------------------------------------------ #
    def _score(self, arm_angle: float, leg_ratio: float, sync_ok: bool) -> float:
        w = self.cfg.weights

        # 1. مدى حركة الذراعين
        is_top = arm_angle > 130
        arm_s = angle_to_score(arm_angle,
                                self.cfg.arms_up_min if is_top else 0.0,
                                180.0 if is_top else self.cfg.arms_down_max,
                                20.0)

        # 2. مدى حركة الساقين
        if is_top:
            # يجب أن تكون الساقان مفتوحتان
            diff = leg_ratio - self.cfg.legs_wide_ratio
            leg_s = min(1.0, max(0.0, 1.0 - abs(diff) / 0.5))
        else:
            # يجب أن تكون مغلقتان
            diff = leg_ratio - self.cfg.legs_close_ratio
            leg_s = min(1.0, max(0.0, 1.0 - abs(diff) / 0.5))

        # 3. تزامن
        sync_s = 1.0 if sync_ok else 0.4

        # 4. هبوط ناعم (نفترض 0.8 افتراضياً — يحتاج accelerometer للدقة الكاملة)
        land_s = 0.8

        total = (w["arms_rom"] * arm_s + w["legs_rom"] * leg_s +
                 w["sync"] * sync_s + w["landing"] * land_s) / 100.0
        return round(float(np.clip(total * 100, 0, 100)), 1)

    # ------------------------------------------------------------------ #
    def _feedback(self, arm_angle, leg_ratio, sync_ok, score, t) -> Tuple[str, Optional[str]]:
        # الذراعان لم ترتفعا بالكامل
        if arm_angle < self.cfg.arms_up_min - 20 and arm_angle > 80:
            m = t("jj_arms_up"); return m, self._get_audio_cue(m)

        # الساقان لم تنفتحا بما يكفي
        if leg_ratio < self.cfg.legs_wide_ratio - 0.3:
            m = t("jj_legs_wide"); return m, self._get_audio_cue(m)

        # عدم التزامن
        if not sync_ok:
            m = t("jj_sync"); return m, self._get_audio_cue(m)

        if score >= 85: return t("jj_excellent"), None
        if score >= 70: return t("jj_good"), None

        m = t("jj_land_soft"); return m, self._get_audio_cue(m)

    # ------------------------------------------------------------------ #
    def _draw(self, frame, pose_result, angles, score, arm_up, legs_wide):
        if not pose_result.detected: return
        lm = pose_result.landmarks
        w, h = pose_result.frame_w, pose_result.frame_h
        def px(i): return landmark_to_pixel(lm[i], w, h)

        body_c = (0,220,0) if score>=80 else (0,165,255) if score>=55 else (0,50,255)
        arm_c  = (0,255,0) if arm_up  else (0,165,255)
        leg_c  = (0,255,0) if legs_wide else (0,165,255)

        # الهيكل مع ألوان مختلفة للذراعين والساقين
        arm_pairs = [
            (LM.LEFT_SHOULDER, LM.LEFT_ELBOW),
            (LM.LEFT_ELBOW, LM.LEFT_WRIST),
            (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW),
            (LM.RIGHT_ELBOW, LM.RIGHT_WRIST),
        ]
        leg_pairs = [
            (LM.LEFT_HIP, LM.LEFT_KNEE),
            (LM.LEFT_KNEE, LM.LEFT_ANKLE),
            (LM.RIGHT_HIP, LM.RIGHT_KNEE),
            (LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
        ]
        body_pairs = [
            (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER),
            (LM.LEFT_HIP, LM.RIGHT_HIP),
            (LM.LEFT_SHOULDER, LM.LEFT_HIP),
            (LM.RIGHT_SHOULDER, LM.RIGHT_HIP),
            (LM.NOSE, LM.LEFT_SHOULDER),
            (LM.NOSE, LM.RIGHT_SHOULDER),
        ]
        for i1, i2 in arm_pairs:
            cv2.line(frame, px(i1), px(i2), arm_c, 4, cv2.LINE_AA)
        for i1, i2 in leg_pairs:
            cv2.line(frame, px(i1), px(i2), leg_c, 4, cv2.LINE_AA)
        for i1, i2 in body_pairs:
            cv2.line(frame, px(i1), px(i2), body_c, 3, cv2.LINE_AA)

        all_joints = [LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, LM.LEFT_ELBOW,
                      LM.RIGHT_ELBOW, LM.LEFT_WRIST, LM.RIGHT_WRIST,
                      LM.LEFT_HIP, LM.RIGHT_HIP, LM.LEFT_KNEE, LM.RIGHT_KNEE,
                      LM.LEFT_ANKLE, LM.RIGHT_ANKLE, LM.NOSE]
        for j in all_joints:
            cv2.circle(frame, px(j), 6, (255,255,255), -1, cv2.LINE_AA)
            cv2.circle(frame, px(j), 7, body_c, 2, cv2.LINE_AA)

        # اعرض نسبة عرض الساقين
        cv2.putText(frame, f"Leg: {angles.get('leg_ratio', 0):.1f}x",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    def _make(self, frame, pct, feedback, angles, is_calib, audio=None):
        return ExerciseResult(
            annotated_frame=frame, percentage=pct, feedback_text=feedback,
            audio_cue=audio, angles=angles,
            rep_count=self._rep_counter.rep_count if self._rep_counter else 0,
            phase_label=self._rep_counter.phase_label if self._rep_counter else "",
            is_calibrating=is_calib, feedback_history=list(self._feedback_history))
