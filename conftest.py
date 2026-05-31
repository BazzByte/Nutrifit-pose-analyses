# =============================================================================
# exercises/pushup_analyzer.py  — v2 (i18n + Arabic)
# =============================================================================

from __future__ import annotations
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from config.thresholds import PUSHUP_CFG, PushupThresholds
from core.angle_utils import calculate_angle, angle_to_score, landmark_to_pixel
from core.pose_detector import PoseResult, MP_LANDMARKS as LM
from core.rep_counter import RepCounter
from exercises.base_exercise import BaseExercise, ExerciseResult


class PushupAnalyzer(BaseExercise):
    """محلل تمرين الضغط / Push-up analyzer."""

    def __init__(self, cfg: PushupThresholds = PUSHUP_CFG, lang: str = "ar") -> None:
        super().__init__(lang=lang)
        self.cfg = cfg
        self._rep_counter = RepCounter(
            top_threshold    = self.cfg.elbow_up_min,
            bottom_threshold = self.cfg.elbow_down_max,
        )

    # ------------------------------------------------------------------ #
    def analyze(self, frame: np.ndarray, pose_result: PoseResult) -> ExerciseResult:
        annotated = frame.copy()
        is_calib  = self._check_calibration()
        t = self.i18n.t   # اختصار للترجمة

        if not pose_result.detected:
            return self._make(annotated, 0.0,
                              f"{t('no_person')} — {t('check_camera')}",
                              {}, is_calib)

        lm = pose_result.landmarks
        w, h = pose_result.frame_w, pose_result.frame_h

        def px(i: int) -> Tuple[int, int]:
            return landmark_to_pixel(lm[i], w, h)

        # ── نقاط ──
        ls = px(LM.LEFT_SHOULDER);  rs = px(LM.RIGHT_SHOULDER)
        le = px(LM.LEFT_ELBOW);     re = px(LM.RIGHT_ELBOW)
        lw = px(LM.LEFT_WRIST);     rw = px(LM.RIGHT_WRIST)
        lh = px(LM.LEFT_HIP);       rh = px(LM.RIGHT_HIP)
        la = px(LM.LEFT_ANKLE);     ra = px(LM.RIGHT_ANKLE)
        nose = px(LM.NOSE)

        # ── زوايا ──
        l_elbow = calculate_angle(ls, le, lw)
        r_elbow = calculate_angle(rs, re, rw)
        elbow   = self._smooth_angle("elbow", (l_elbow + r_elbow) / 2)

        l_hip_a = calculate_angle(ls, lh, la)
        r_hip_a = calculate_angle(rs, rh, ra)
        hip     = self._smooth_angle("hip", (l_hip_a + r_hip_a) / 2)

        neck    = self._smooth_angle("neck", calculate_angle(lh, ls, nose))

        angles: Dict[str, float] = {
            "elbow": round(elbow, 1), "hip": round(hip, 1),
            "neck":  round(neck, 1),  "l_elbow": round(l_elbow, 1),
        }

        # ── تكرارات ──
        self._rep_counter.update(elbow)

        # ── نسبة ──
        pct = self._score(elbow, hip, neck) if not is_calib else (
            angle_to_score(hip, self.cfg.back_straight_min, self.cfg.back_straight_max, 20) * 100
        )

        feedback, audio = self._feedback(elbow, hip, neck, pct, t)
        self._add_feedback(feedback)
        self._draw(annotated, pose_result, angles, pct)

        return self._make(annotated, pct, feedback, angles, is_calib, audio)

    # ------------------------------------------------------------------ #
    def _score(self, elbow: float, hip: float, neck: float) -> float:
        w = self.cfg.weights
        is_bottom = self._rep_counter._reached_bottom

        elbow_s = angle_to_score(elbow,
                                  self.cfg.elbow_down_min if is_bottom else self.cfg.elbow_up_min,
                                  self.cfg.elbow_down_max if is_bottom else self.cfg.elbow_up_max,
                                  20.0)
        back_s  = angle_to_score(hip, self.cfg.back_straight_min, self.cfg.back_straight_max, 15.0)
        hip_s   = (1.0 if self.cfg.hip_sag_threshold <= hip <= self.cfg.hip_pike_threshold + 15
                   else max(0.0, (hip - 130) / 35.0))
        neck_s  = angle_to_score(neck, self.cfg.neck_alignment_min, 180.0, 20.0)

        total = (w["elbow_rom"] * elbow_s + w["back_straight"] * back_s +
                 w["hip_position"] * hip_s + w["neck_align"] * neck_s +
                 w["descent_speed"] * 0.8) / 100.0
        return round(float(np.clip(total * 100, 0, 100)), 1)

    # ------------------------------------------------------------------ #
    def _feedback(self, elbow, hip, neck, score, t) -> Tuple[str, Optional[str]]:
        if hip < self.cfg.hip_sag_threshold - 10:
            m = t("pu_sagging"); return m, self._get_audio_cue(m)
        if hip > 175:
            m = t("pu_piking"); return m, self._get_audio_cue(m)
        if neck < self.cfg.neck_alignment_min - 15:
            m = t("pu_head_down"); return m, self._get_audio_cue(m)
        if not self._rep_counter._reached_bottom and 100 < elbow < 155:
            m = t("pu_go_deeper"); return m, self._get_audio_cue(m)
        if elbow < self.cfg.elbow_up_min - 10 and score < 70:
            m = t("pu_extend_arms"); return m, self._get_audio_cue(m)
        if score >= 85: return t("pu_excellent"), None
        if score >= 70: return t("pu_good"), None
        m = t("pu_slow_descent"); return m, self._get_audio_cue(m)

    # ------------------------------------------------------------------ #
    def _draw(self, frame, pose_result, angles, score):
        if not pose_result.detected: return
        lm = pose_result.landmarks
        w, h = pose_result.frame_w, pose_result.frame_h
        def px(i): return landmark_to_pixel(lm[i], w, h)

        c = (0,220,0) if score>=80 else (0,165,255) if score>=55 else (0,50,255)
        pairs = [
            (LM.LEFT_SHOULDER, LM.LEFT_ELBOW, 4),
            (LM.LEFT_ELBOW, LM.LEFT_WRIST, 4),
            (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, 4),
            (LM.RIGHT_ELBOW, LM.RIGHT_WRIST, 4),
            (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, 3),
            (LM.LEFT_HIP, LM.RIGHT_HIP, 3),
            (LM.LEFT_SHOULDER, LM.LEFT_HIP, 4),
            (LM.RIGHT_SHOULDER, LM.RIGHT_HIP, 4),
            (LM.LEFT_HIP, LM.LEFT_KNEE, 3),
            (LM.LEFT_KNEE, LM.LEFT_ANKLE, 3),
            (LM.RIGHT_HIP, LM.RIGHT_KNEE, 3),
            (LM.RIGHT_KNEE, LM.RIGHT_ANKLE, 3),
            (LM.NOSE, LM.LEFT_SHOULDER, 2),
            (LM.NOSE, LM.RIGHT_SHOULDER, 2),
        ]
        for i1, i2, t in pairs:
            cv2.line(frame, px(i1), px(i2), c, t, cv2.LINE_AA)
        for j in [LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, LM.LEFT_ELBOW,
                  LM.RIGHT_ELBOW, LM.LEFT_WRIST, LM.RIGHT_WRIST,
                  LM.LEFT_HIP, LM.RIGHT_HIP, LM.LEFT_KNEE, LM.RIGHT_KNEE,
                  LM.LEFT_ANKLE, LM.RIGHT_ANKLE, LM.NOSE]:
            cv2.circle(frame, px(j), 6, (255,255,255), -1, cv2.LINE_AA)
            cv2.circle(frame, px(j), 7, c, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{int(angles.get('elbow',0))}°",
                    (px(LM.LEFT_ELBOW)[0]+10, px(LM.LEFT_ELBOW)[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    def _make(self, frame, pct, feedback, angles, is_calib, audio=None):
        return ExerciseResult(
            annotated_frame  = frame,
            percentage       = pct,
            feedback_text    = feedback,
            audio_cue        = audio,
            angles           = angles,
            rep_count        = self._rep_counter.rep_count if self._rep_counter else 0,
            phase_label      = self._rep_counter.phase_label if self._rep_counter else "",
            is_calibrating   = is_calib,
            feedback_history = list(self._feedback_history),
        )
