# =============================================================================
# exercises/squat_analyzer.py
# محلل تمرين الجلسة (Squat)
# Squat exercise analyzer — chosen as the 4th exercise
#
# اختيار Squat لأنه:
# ✓ يعمل مع كاميرا أمامية أو جانبية
# ✓ يُكمِّل المجموعة (جزء علوي + علقة + قلب + بطن + جزء سفلي)
# ✓ أكثر تمارين الجسم السفلي شيوعاً
# =============================================================================

from __future__ import annotations
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from config.thresholds import SQUAT_CFG, SquatThresholds
from core.angle_utils import calculate_angle, angle_to_score, landmark_to_pixel, euclidean_distance
from core.pose_detector import PoseResult, MP_LANDMARKS as LM
from core.rep_counter import RepCounter
from exercises.base_exercise import BaseExercise, ExerciseResult


class SquatAnalyzer(BaseExercise):
    """
    محلل تمرين الجلسة (Squat).
    Squat analyzer — best results with side-view camera.

    يقيّم:
    ✓ زاوية الركبة (Knee angle ROM)
    ✓ ميل الظهر (Back lean angle)
    ✓ انهيار الركبتين للداخل (Knee valgus)
    ✓ ثبات الكعبين (Heel stability)
    ✓ عمق الجلسة (Depth — thighs parallel to floor)
    ✓ حساب التكرارات
    """

    def __init__(self, cfg: SquatThresholds = SQUAT_CFG, lang: str = "ar") -> None:
        super().__init__(lang=lang)
        self.cfg = cfg

        # زاوية الركبة: top=واقف(170°), bottom=جالس(90°)
        self._rep_counter = RepCounter(
            top_threshold    = self.cfg.knee_up_min,
            bottom_threshold = self.cfg.knee_down_max,
        )

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

        # ── زاوية الركبة (الورك - الركبة - الكاحل) ──
        l_knee_a = calculate_angle(lh, lk, la)
        r_knee_a = calculate_angle(rh, rk, ra)
        knee_a   = self._smooth_angle("knee", (l_knee_a + r_knee_a) / 2)

        # ── ميل الظهر: الزاوية العمودية للكتف-الورك ──
        # نقيس الزاوية بين خط الكتف-الورك والمحور الرأسي
        # نستخدم الزاوية (أنف - كتف - ورك) كبديل
        back_a   = self._smooth_angle("back", calculate_angle(nose, ls, lh))

        # ── انهيار الركبتين للداخل (Knee Valgus) ──
        # نقارن عرض الركبتين بعرض الكاحلين
        knee_w  = euclidean_distance(lk, rk)
        ankle_w = euclidean_distance(la, ra)
        # إذا كانت الركبتان أضيق من الكاحلين = انهيار داخلي
        valgus_ratio = (knee_w / ankle_w) if ankle_w > 0 else 1.0
        valgus_ok    = valgus_ratio >= 0.85   # الركبتان على الأقل 85% من عرض الكاحلين

        # ── ثبات الكعب (نُقدّر من ارتفاع الكاحل) ──
        # إذا الكاحل ارتفع كثيراً عن الأرض = رفع الكعب
        heel_y_avg  = (la[1] + ra[1]) / 2
        hip_y_avg   = (lh[1] + rh[1]) / 2
        heel_stable = (hip_y_avg - heel_y_avg) > (h * 0.1)   # تقريبي

        angles: Dict[str, float] = {
            "knee":         round(knee_a, 1),
            "back":         round(back_a, 1),
            "valgus_ratio": round(valgus_ratio, 2),
            "l_knee":       round(l_knee_a, 1),
            "r_knee":       round(r_knee_a, 1),
        }

        # ── تكرارات ──
        self._rep_counter.update(knee_a)

        # ── نسبة ──
        pct = self._score(knee_a, back_a, valgus_ok, heel_stable) if not is_calib else (
            angle_to_score(knee_a, self.cfg.knee_up_min, self.cfg.knee_up_max, 15.0) * 100
        )

        feedback, audio = self._feedback(knee_a, back_a, valgus_ok, heel_stable, pct, t)
        self._add_feedback(feedback)
        self._draw(annotated, pose_result, angles, pct, valgus_ok)

        return self._make(annotated, pct, feedback, angles, is_calib, audio)

    # ------------------------------------------------------------------ #
    def _score(self, knee_a, back_a, valgus_ok, heel_stable):
        w = self.cfg.weights
        is_bottom = self._rep_counter._reached_bottom

        # 1. مدى حركة الركبة
        if is_bottom:
            knee_s = angle_to_score(knee_a, self.cfg.knee_down_min,
                                     self.cfg.knee_down_max, 20.0)
        else:
            knee_s = angle_to_score(knee_a, self.cfg.knee_up_min,
                                     self.cfg.knee_up_max, 15.0)

        # 2. زاوية الظهر (لا ينحني كثيراً للأمام)
        back_s = angle_to_score(back_a, 90.0, 180.0 - self.cfg.back_lean_max, 20.0)

        # 3. تتبع الركبة (لا تنهار للداخل)
        knee_track_s = 1.0 if valgus_ok else 0.3

        # 4. ثبات الكعب
        heel_s = 0.9 if heel_stable else 0.5

        total = (w["knee_rom"] * knee_s + w["back_angle"] * back_s +
                 w["knee_track"] * knee_track_s + w["heel_down"] * heel_s) / 100.0
        return round(float(np.clip(total * 100, 0, 100)), 1)

    # ------------------------------------------------------------------ #
    def _feedback(self, knee_a, back_a, valgus_ok, heel_stable, score, t) -> Tuple[str, Optional[str]]:
        # الركبتان تنهاران
        if not valgus_ok:
            m = t("sq_knees_out"); return m, self._get_audio_cue(m)

        # الظهر ينحني للأمام كثيراً
        if back_a < 90.0 - self.cfg.back_lean_max:
            m = t("sq_back_straight"); return m, self._get_audio_cue(m)

        # لم ينزل بعمق كافٍ
        if knee_a > self.cfg.knee_down_max + 25 and self._rep_counter._reached_bottom:
            m = t("sq_deeper"); return m, self._get_audio_cue(m)

        # رفع الكعب
        if not heel_stable:
            m = t("sq_heels_up"); return m, self._get_audio_cue(m)

        # لم يمدد الركبتين بالكامل في الأعلى
        if knee_a < self.cfg.knee_up_min - 10 and not self._rep_counter._reached_bottom:
            m = t("sq_extend_top"); return m, self._get_audio_cue(m)

        if score >= 85: return t("sq_excellent"), None
        if score >= 70: return t("sq_good"), None
        m = t("sq_deeper"); return m, self._get_audio_cue(m)

    # ------------------------------------------------------------------ #
    def _draw(self, frame, pose_result, angles, score, valgus_ok):
        if not pose_result.detected: return
        lm = pose_result.landmarks
        w, h = pose_result.frame_w, pose_result.frame_h
        def px(i): return landmark_to_pixel(lm[i], w, h)

        body_c = (0,220,0) if score>=80 else (0,165,255) if score>=55 else (0,50,255)
        knee_c = (0,220,0) if valgus_ok else (0,50,255)

        # الجزء العلوي
        upper_pairs = [
            (LM.NOSE, LM.LEFT_SHOULDER, body_c, 2),
            (LM.NOSE, LM.RIGHT_SHOULDER, body_c, 2),
            (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, body_c, 3),
            (LM.LEFT_SHOULDER, LM.LEFT_HIP, body_c, 4),
            (LM.RIGHT_SHOULDER, LM.RIGHT_HIP, body_c, 4),
            (LM.LEFT_HIP, LM.RIGHT_HIP, body_c, 3),
        ]
        # الجزء السفلي (ألوان الركبة)
        lower_pairs = [
            (LM.LEFT_HIP, LM.LEFT_KNEE, knee_c, 4),
            (LM.LEFT_KNEE, LM.LEFT_ANKLE, knee_c, 4),
            (LM.RIGHT_HIP, LM.RIGHT_KNEE, knee_c, 4),
            (LM.RIGHT_KNEE, LM.RIGHT_ANKLE, knee_c, 4),
        ]
        for i1, i2, c, t in upper_pairs + lower_pairs:
            cv2.line(frame, px(i1), px(i2), c, t, cv2.LINE_AA)

        all_joints = [LM.NOSE, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
                      LM.LEFT_HIP, LM.RIGHT_HIP, LM.LEFT_KNEE, LM.RIGHT_KNEE,
                      LM.LEFT_ANKLE, LM.RIGHT_ANKLE]
        for j in all_joints:
            jc = knee_c if j in [LM.LEFT_KNEE, LM.RIGHT_KNEE] else body_c
            cv2.circle(frame, px(j), 6, (255,255,255), -1, cv2.LINE_AA)
            cv2.circle(frame, px(j), 7, jc, 2, cv2.LINE_AA)

        # اعرض زاوية الركبة
        cv2.putText(frame, f"Knee: {int(angles.get('knee',0))}°",
                    (px(LM.LEFT_KNEE)[0]+10, px(LM.LEFT_KNEE)[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2, cv2.LINE_AA)
        # مؤشر انهيار الركبة
        if not valgus_ok:
            cv2.putText(frame, "KNEE IN!",
                        (px(LM.LEFT_KNEE)[0]-20, px(LM.LEFT_KNEE)[1]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,50,255), 2, cv2.LINE_AA)

    def _make(self, frame, pct, feedback, angles, is_calib, audio=None):
        return ExerciseResult(
            annotated_frame=frame, percentage=pct, feedback_text=feedback,
            audio_cue=audio, angles=angles,
            rep_count=self._rep_counter.rep_count if self._rep_counter else 0,
            phase_label=self._rep_counter.phase_label if self._rep_counter else "",
            is_calibrating=is_calib, feedback_history=list(self._feedback_history))
