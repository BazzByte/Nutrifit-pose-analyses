# =============================================================================
# exercises/base_exercise.py
# الكلاس الأساسي المجرد — محدَّث بدعم اللغتين والخط العربي
# Abstract base — updated with i18n and Arabic renderer support
# =============================================================================

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np

from core.rep_counter import RepCounter
from visualization.i18n import I18n, i18n as _global_i18n


@dataclass
class ExerciseResult:
    """نتيجة frame واحد / Single-frame analysis result."""
    annotated_frame:  np.ndarray
    percentage:       float
    feedback_text:    str
    audio_cue:        Optional[str]
    angles:           Dict[str, float]
    rep_count:        int
    phase_label:      str
    is_calibrating:   bool
    feedback_history: List[str]


class BaseExercise(ABC):
    """
    الكلاس الأساسي لكل تمرين.
    Abstract base for all exercise analyzers.
    """

    CALIBRATION_DURATION: float = 2.5   # ثواني

    def __init__(self, lang: str = "ar") -> None:
        # اللغة
        self.i18n: I18n = I18n(lang=lang)  # type: ignore

        self._start_time:     float = time.time()
        self._is_calibrating: bool  = True

        self._feedback_history: Deque[str] = deque(maxlen=4)
        self._angle_history:    Dict[str, list] = {}
        self._rep_counter:      Optional[RepCounter] = None

        self._last_audio_cue: Optional[str] = None
        self._last_audio_ts:  float = 0.0
        self._audio_interval: float = 4.0

    # ------------------------------------------------------------------ #
    @abstractmethod
    def analyze(self, frame: np.ndarray, pose_result) -> ExerciseResult:
        """حلّل frame واحد / Analyze a single frame."""
        ...

    # ------------------------------------------------------------------ #
    def set_language(self, lang: str) -> None:
        """غيّر اللغة في أثناء التشغيل / Change language at runtime."""
        self.i18n.set_lang(lang)  # type: ignore

    def _check_calibration(self) -> bool:
        elapsed = time.time() - self._start_time
        if elapsed < self.CALIBRATION_DURATION:
            return True
        self._is_calibrating = False
        return False

    def _add_feedback(self, feedback: str) -> None:
        if not self._feedback_history or self._feedback_history[0] != feedback:
            self._feedback_history.appendleft(feedback)

    def _get_audio_cue(self, cue: str) -> Optional[str]:
        now = time.time()
        if cue != self._last_audio_cue or (now - self._last_audio_ts) > self._audio_interval:
            self._last_audio_cue = cue
            self._last_audio_ts  = now
            return cue
        return None

    def _smooth_angle(self, key: str, value: float, window: int = 5) -> float:
        if key not in self._angle_history:
            self._angle_history[key] = []
        buf = self._angle_history[key]
        buf.append(value)
        if len(buf) > window:
            buf.pop(0)
        return float(np.mean(buf))

    def reset(self) -> None:
        self._start_time      = time.time()
        self._is_calibrating  = True
        self._feedback_history.clear()
        self._angle_history.clear()
        if self._rep_counter:
            self._rep_counter.reset()
