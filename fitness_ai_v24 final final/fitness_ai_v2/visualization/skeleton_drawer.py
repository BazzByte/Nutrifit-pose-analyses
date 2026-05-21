# =============================================================================
# visualization/skeleton_drawer.py  — v2
# رسم HUD الكامل مع دعم العربية
# Full HUD renderer with Arabic text support via Pillow
# =============================================================================

from __future__ import annotations
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config.thresholds import VIZ_CFG, VisualizationConfig
from exercises.base_exercise import ExerciseResult
from visualization.arabic_renderer import renderer as ar
from visualization.i18n import I18n, i18n as _global_i18n


# اسم كل تمرين بالعربي والإنجليزي
_EXERCISE_NAMES = {
    "pushup":      {"ar": "تمرين الضغط",     "en": "Push-ups"},
    "pullup":      {"ar": "تمرين العقلة",    "en": "Pull-ups"},
    "jumpingjack": {"ar": "القفز المتفرج",   "en": "Jumping Jacks"},
    "crunch":      {"ar": "تمرين البطن",     "en": "Crunches"},
    "squat":       {"ar": "تمرين الجلسة",   "en": "Squats"},
}


class HUDRenderer:
    """
    رسم طبقة HUD فوق الـ frame.
    Draws the HUD overlay on top of the annotated frame.

    ┌────────────────────────────────────────┐
    │  [شريط علوي] التعليق التصحيحي          │
    │  [يمين] تاريخ التعليقات               │
    │                                        │
    │        [الهيكل العظمي الملون]          │
    │                                        │
    │  [شريط أسفل] رقم التكرار | % | مرحلة │
    │  [شريط تقدم ملون]                      │
    └────────────────────────────────────────┘
    """

    def __init__(self, cfg: VisualizationConfig = VIZ_CFG,
                 i18n: I18n = _global_i18n) -> None:
        self.cfg  = cfg
        self.i18n = i18n

    # ------------------------------------------------------------------ #
    def render(self,
               result: ExerciseResult,
               exercise_type: str,
               fps: float = 0.0) -> np.ndarray:
        """
        أضف HUD على الـ frame وأرجع الإطار النهائي.
        Apply HUD to annotated frame and return final display frame.
        """
        frame = result.annotated_frame.copy()
        h, w  = frame.shape[:2]
        t     = self.i18n.t

        # 1. شريط علوي: التعليق
        frame = self._draw_top_bar(frame, result.feedback_text, w, h)

        # 2. شريط سفلي: تكرارات + نسبة + مرحلة
        frame = self._draw_bottom_bar(frame, result, exercise_type, w, h, t)

        # 3. شريط التقدم
        self._draw_progress_bar(frame, result.percentage, w, h)

        # 4. تاريخ التعليقات
        frame = self._draw_history(frame, result.feedback_history, w, h)

        # 5. FPS
        if fps > 0:
            cv2.putText(frame, f"FPS:{fps:.0f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160,160,160), 1, cv2.LINE_AA)

        # 6. Calibration overlay
        if result.is_calibrating:
            frame = self._draw_calibration(frame, w, h, t)

        return frame

    # ------------------------------------------------------------------ #
    def _draw_top_bar(self, frame, feedback, w, h):
        bar_h = 60
        # خلفية شبه شفافة
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (0, bar_h-1), (w, bar_h), (60,60,60), 1)

        # النص — ارسم بالوسط
        font_sz = self.cfg.font_size_medium
        tw, th  = ar.get_text_size(feedback, font_sz)
        x = max(8, (w - tw) // 2)
        y = (bar_h - th) // 2
        frame = ar.put_text(frame, feedback, (x, y), font_sz,
                            color=(255,255,255), bold=True)
        return frame

    # ------------------------------------------------------------------ #
    def _draw_bottom_bar(self, frame, result, exercise_type, w, h, t):
        bar_h  = 90
        bar_y  = h - bar_h
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, bar_y), (w, h), (15,15,15), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
        cv2.rectangle(frame, (0, bar_y), (w, bar_y+1), (60,60,60), 1)

        lang   = self.i18n.lang
        ex_label = _EXERCISE_NAMES.get(exercise_type, {}).get(lang, exercise_type)
        reps_label = t("reps")

        # ── اسم التمرين + تكرارات (يسار) ──
        frame = ar.put_text(frame, ex_label, (12, bar_y + 8),
                            self.cfg.font_size_small, (180,180,180))
        frame = ar.put_text(frame, f"{reps_label}: {result.rep_count}",
                            (12, bar_y + 36),
                            self.cfg.font_size_large + 4, (100,255,100), bold=True)

        # ── نسبة الصحة (وسط) ──
        pct_color = self._pct_color(result.percentage)
        score_label = t("form_score")
        pct_str     = f"{result.percentage:.0f}%"

        tw, _ = ar.get_text_size(score_label, self.cfg.font_size_small)
        frame = ar.put_text(frame, score_label,
                            ((w - tw) // 2, bar_y + 8),
                            self.cfg.font_size_small, (180,180,180))

        tw, _ = ar.get_text_size(pct_str, self.cfg.font_size_large + 10)
        frame = ar.put_text(frame, pct_str,
                            ((w - tw) // 2, bar_y + 32),
                            self.cfg.font_size_large + 10,
                            pct_color, bold=True)

        # ── المرحلة (يمين) ──
        phase_str = result.phase_label
        tw, _  = ar.get_text_size(phase_str, self.cfg.font_size_medium)
        frame  = ar.put_text(frame, phase_str,
                             (w - tw - 12, bar_y + 36),
                             self.cfg.font_size_medium, (0,200,255))
        return frame

    # ------------------------------------------------------------------ #
    def _draw_progress_bar(self, frame, pct, w, h):
        bar_y   = h - 95
        bar_x   = 10
        max_w   = w - 20
        bar_ht  = 8
        fill_w  = int(max_w * pct / 100.0)

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + max_w, bar_y + bar_ht),
                      (50,50,50), -1)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + fill_w, bar_y + bar_ht),
                          self._pct_color(pct), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + max_w, bar_y + bar_ht),
                      (100,100,100), 1)

    # ------------------------------------------------------------------ #
    def _draw_history(self, frame, history, w, h):
        if not history: return frame
        x_start = w - 300
        y_start = 70

        for i, msg in enumerate(history[:4]):
            alpha = max(0.25, 1.0 - i * 0.25)
            c = tuple(int(v * alpha) for v in (190, 190, 190))
            sz = max(12, self.cfg.font_size_small - i * 2)
            display = (msg[:30] + "..") if len(msg) > 32 else msg
            frame = ar.put_text(frame, display,
                                (x_start, y_start + i * 28),
                                sz, c)
        return frame

    # ------------------------------------------------------------------ #
    def _draw_calibration(self, frame, w, h, t):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.40, frame, 0.60, 0, frame)

        calib_text = t("calibrating")
        tw, th = ar.get_text_size(calib_text, self.cfg.font_size_large + 4)
        frame = ar.put_text(frame, calib_text,
                            ((w-tw)//2, h//2 - 30),
                            self.cfg.font_size_large + 4,
                            (0,255,255), bold=True)

        sub_text = t("hold_start_pos")
        tw2, _ = ar.get_text_size(sub_text, self.cfg.font_size_medium)
        frame = ar.put_text(frame, sub_text,
                            ((w-tw2)//2, h//2 + 10),
                            self.cfg.font_size_medium, (200,200,200))
        return frame

    # ------------------------------------------------------------------ #
    @staticmethod
    def _pct_color(pct: float) -> Tuple[int,int,int]:
        if pct >= 80: return (0, 220, 50)
        if pct >= 55: return (0, 165, 255)
        return (0, 50, 255)
