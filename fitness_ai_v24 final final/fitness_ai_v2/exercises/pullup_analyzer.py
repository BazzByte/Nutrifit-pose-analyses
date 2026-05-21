# =============================================================================
# exercises/pullup_analyzer.py  — v2 (i18n + Arabic)
# =============================================================================

from __future__ import annotations
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from config.thresholds import PULLUP_CFG, PullupThresholds
from core.angle_utils import calculate_angle, angle_to_score, landmark_to_pixel
from core.pose_detector import PoseResult, MP_LANDMARKS as LM
from core.rep_counter import RepCounter
from exercises.base_exercise import BaseExercise, ExerciseResult


class PullupAnalyzer(BaseExercise):
    """محلل تمرين العقلة / Pull-up analyzer."""

    def __init__(self, cfg: PullupThresholds = PULLUP_CFG, lang: str = "ar") -> None:
        super().__init__(lang=lang)
        self.cfg = cfg
        self._rep_counter = RepCounter(
            top_threshold    = self.cfg.elbow_top_max,
            bottom_threshold = self.cfg.elbow_bottom_min,
        )
        self._hip_x_history: list[float] = []

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

        ls = px(LM.LEFT_SHOULDER);  rs = px(LM.RIGHT_SHOULDER)
        le = px(LM.LEFT_ELBOW);     re = px(LM.RIGHT_ELBOW)
        lw = px(LM.LEFT_WRIST);     rw = px(LM.RIGHT_WRIST)
        lh = px(LM.LEFT_HIP);       rh = px(LM.RIGHT_HIP)

        l_elbow_a = calculate_angle(ls, le, lw)
        r_elbow_a = calculate_angle(rs, re, rw)
        elbow     = self._smooth_angle("elbow", (l_elbow_a + r_elbow_a) / 2)

        l_sh_a    = calculate_angle(le, ls, lh)
        r_sh_a    = calculate_angle(re, rs, rh)
        shoulder  = self._smooth_angle("shoulder", (l_sh_a + r_sh_a) / 2)

        hip_cx = (lh[0] + rh[0]) / 2.0
        self._hip_x_history.append(hip_cx)
        if len(self._hip_x_history) > 30: self._hip_x_history.pop(0)
        swing_score = 1.0
        if len(self._hip_x_history) >= 10:
            swing_deg = (float(np.std(self._hip_x_history)) / w) * 180.0
            if swing_deg > self.cfg.swing_tolerance_deg:
                swing_score = max(0.0, 1.0 - (swing_deg - self.cfg.swing_tolerance_deg) / 30.0)

        angles = {"elbow": round(elbow,1), "shoulder": round(shoulder,1),
                  "l_elbow": round(l_elbow_a,1), "swing_score": round(swing_score,2)}

        self._rep_counter.update(180.0 - elbow)

        pct = (angle_to_score(elbow, 155.0, 180.0, 20.0) * 100 if is_calib
               else self._score(elbow, shoulder, swing_score))

        feedback, audio = self._feedback(elbow, shoulder, swing_score, pct, t)
        self._add_feedback(feedback)
        self._draw(annotated, pose_result, angles, pct)
        return self._make(annotated, pct, feedback, angles, is_calib, audio)

    # ------------------------------------------------------------------ #
    def _score(self, elbow, shoulder, swing_score):
        w = self.cfg.weights
        is_bottom = not self._rep_counter._reached_bottom
        elbow_s   = angle_to_score(elbow,
                                    self.cfg.elbow_bottom_min if is_bottom else self.cfg.elbow_top_min,
                                    self.cfg.elbow_bottom_max if is_bottom else self.cfg.elbow_top_max,
                                    15.0)
        sh_s      = angle_to_score(shoulder, self.cfg.shoulder_retract_threshold, 90.0, 20.0)
        full_s    = angle_to_score(elbow, 155.0, 180.0, 20.0) if is_bottom else 0.9
        total = (w["elbow_rom"] * elbow_s + w["shoulder_retract"] * sh_s +
                 w["no_swing"] * swing_score + w["full_extension"] * full_s) / 100.0
        return round(float(np.clip(total * 100, 0, 100)), 1)

    # ------------------------------------------------------------------ #
    def _feedback(self, elbow, shoulder, swing_score, score, t):
        if swing_score < 0.5:
            m = t("plu_swinging"); return m, self._get_audio_cue(m)
        if elbow > self.cfg.elbow_top_max + 20:
            m = t("plu_go_higher"); return m, self._get_audio_cue(m)
        if elbow < self.cfg.elbow_bottom_min - 15:
            m = t("plu_full_hang"); return m, self._get_audio_cue(m)
        if shoulder < self.cfg.shoulder_retract_threshold - 10:
            m = t("plu_retract"); return m, self._get_audio_cue(m)
        if swing_score < 0.75:
            m = t("plu_reduce_swing"); return m, self._get_audio_cue(m)
        if score >= 85: return t("plu_excellent"), None
        if score >= 70: return t("plu_good"), None
        m = t("plu_slow_descent"); return m, self._get_audio_cue(m)

    # ------------------------------------------------------------------ #
    def _draw(self, frame, pose_result, angles, score):
        if not pose_result.detected: return
        lm = pose_result.landmarks
        w, h = pose_result.frame_w, pose_result.frame_h
        def px(i): return landmark_to_pixel(lm[i], w, h)
        c = (0,220,0) if score>=80 else (0,165,255) if score>=55 else (0,50,255)
        pairs = [
            (LM.LEFT_SHOULDER, LM.LEFT_ELBOW, 5),
            (LM.LEFT_ELBOW, LM.LEFT_WRIST, 5),
            (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, 5),
            (LM.RIGHT_ELBOW, LM.RIGHT_WRIST, 5),
            (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, 3),
            (LM.LEFT_SHOULDER, LM.LEFT_HIP, 3),
            (LM.RIGHT_SHOULDER, LM.RIGHT_HIP, 3),
            (LM.LEFT_HIP, LM.RIGHT_HIP, 3),
            (LM.NOSE, LM.LEFT_SHOULDER, 2),
            (LM.NOSE, LM.RIGHT_SHOULDER, 2),
        ]
        for i1, i2, t in pairs:
            cv2.line(frame, px(i1), px(i2), c, t, cv2.LINE_AA)
        for j in [LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, LM.LEFT_ELBOW,
                  LM.RIGHT_ELBOW, LM.LEFT_WRIST, LM.RIGHT_WRIST,
                  LM.LEFT_HIP, LM.RIGHT_HIP, LM.NOSE]:
            cv2.circle(frame, px(j), 7, (255,255,255), -1, cv2.LINE_AA)
            cv2.circle(frame, px(j), 8, c, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{int(angles.get('elbow',0))}°",
                    (px(LM.LEFT_ELBOW)[0]+10, px(LM.LEFT_ELBOW)[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2, cv2.LINE_AA)

    def _make(self, frame, pct, feedback, angles, is_calib, audio=None):
        return ExerciseResult(
            annotated_frame=frame, percentage=pct, feedback_text=feedback,
            audio_cue=audio, angles=angles,
            rep_count=self._rep_counter.rep_count if self._rep_counter else 0,
            phase_label=self._rep_counter.phase_label if self._rep_counter else "",
            is_calibrating=is_calib, feedback_history=list(self._feedback_history))
