# =============================================================================
# core/rep_counter.py
# نظام حساب التكرارات الدقيق
# Accurate repetition counting system with state machine
# =============================================================================

import time
from enum import Enum, auto
from typing import Optional

from config.thresholds import REP_CFG, RepCounterConfig


class RepPhase(Enum):
    """مراحل التكرار / Phases of a single repetition."""
    IDLE       = auto()   # في انتظار البدء
    GOING_DOWN = auto()   # في طريق النزول / going to bottom
    BOTTOM     = auto()   # وصل الأسفل / at bottom position
    GOING_UP   = auto()   # في طريق الصعود / going to top
    TOP        = auto()   # وصل الأعلى / at top position


class RepCounter:
    """
    حاسبة تكرارات تعتمد على آلة الحالة (State Machine).
    State-machine-based repetition counter.

    الفكرة:
    - نتتبع زاوية الكوع (أو أي زاوية رئيسية)
    - الحركة: TOP → BOTTOM → TOP = تكرار واحد صحيح
    Idea:
    - Track primary angle (elbow angle typically)
    - Full cycle: TOP → BOTTOM → TOP = 1 complete rep
    """

    def __init__(self,
                 cfg: RepCounterConfig = REP_CFG,
                 top_threshold: float = 155.0,
                 bottom_threshold: float = 100.0) -> None:
        """
        Args:
            cfg:               إعدادات الحساب
            top_threshold:     الزاوية الدالة على الوضعية العلوية (مفتوح)
            bottom_threshold:  الزاوية الدالة على الوضعية السفلية (مغلق)
        """
        self.cfg              = cfg
        self.top_threshold    = top_threshold
        self.bottom_threshold = bottom_threshold

        self.rep_count:    int       = 0
        self.phase:        RepPhase  = RepPhase.IDLE
        self._confirm_buf: int       = 0
        self._last_rep_ts: float     = 0.0
        self._reached_bottom: bool   = False

    # ------------------------------------------------------------------ #
    def update(self, angle: float) -> bool:
        """
        أعطِ الزاوية الحالية واحصل على True إذا اكتمل تكرار جديد.
        Feed current angle; returns True when a new rep is completed.

        Args:
            angle: زاوية الكوع (أو الزاوية الأساسية للتمرين)

        Returns:
            True إذا تم إكمال تكرار كامل في هذا الـ frame
        """
        now = time.time()

        # --- الوضعية العلوية (زاوية كبيرة = ذراع مفتوحة)
        if angle >= self.top_threshold:
            if self._reached_bottom:
                # اكتمل تكرار: كان في الأسفل والآن رجع للأعلى
                self._confirm_buf += 1
                if self._confirm_buf >= self.cfg.confirmation_frames:
                    if (now - self._last_rep_ts) >= self.cfg.min_rep_interval:
                        self.rep_count     += 1
                        self._last_rep_ts   = now
                        self._reached_bottom = False
                        self._confirm_buf    = 0
                        self.phase           = RepPhase.TOP
                        return True   # ← تكرار مكتمل!
            else:
                self.phase        = RepPhase.TOP
                self._confirm_buf = 0

        # --- الوضعية السفلية (زاوية صغيرة = ذراع منثنية)
        elif angle <= self.bottom_threshold:
            self._reached_bottom = True
            self._confirm_buf    = 0
            self.phase           = RepPhase.BOTTOM

        # --- في المنتصف
        else:
            self._confirm_buf = 0
            if self.phase == RepPhase.TOP:
                self.phase = RepPhase.GOING_DOWN
            elif self.phase == RepPhase.BOTTOM:
                self.phase = RepPhase.GOING_UP

        return False

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """إعادة تعيين العداد / Reset the counter."""
        self.rep_count       = 0
        self.phase           = RepPhase.IDLE
        self._confirm_buf    = 0
        self._last_rep_ts    = 0.0
        self._reached_bottom = False

    # ------------------------------------------------------------------ #
    @property
    def phase_label(self) -> str:
        """نص الحالة الحالية للعرض / Human-readable current phase."""
        labels = {
            RepPhase.IDLE:       "جاهز",
            RepPhase.GOING_DOWN: "نزول ↓",
            RepPhase.BOTTOM:     "أدنى نقطة",
            RepPhase.GOING_UP:   "صعود ↑",
            RepPhase.TOP:        "أعلى نقطة",
        }
        return labels.get(self.phase, "")
