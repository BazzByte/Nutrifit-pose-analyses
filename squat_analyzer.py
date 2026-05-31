# =============================================================================
# config/thresholds.py
# عتبات وإعدادات كل التمارين الستة
# Thresholds & config for all 6 exercises
# =============================================================================

from dataclasses import dataclass, field
from typing import Dict, Tuple


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PushupThresholds:
    """تمرين الضغط / Push-up"""
    elbow_down_min:       float = 70.0
    elbow_down_max:       float = 100.0
    elbow_up_min:         float = 155.0
    elbow_up_max:         float = 180.0
    back_straight_min:    float = 160.0
    back_straight_max:    float = 180.0
    hip_pike_threshold:   float = 155.0
    hip_sag_threshold:    float = 165.0
    neck_alignment_min:   float = 150.0
    weights: Dict[str, float] = field(default_factory=lambda: {
        "elbow_rom":     30.0,
        "back_straight": 30.0,
        "hip_position":  20.0,
        "neck_align":    10.0,
        "descent_speed": 10.0,
    })


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PullupThresholds:
    """تمرين العقلة / Pull-up"""
    elbow_top_min:              float = 30.0
    elbow_top_max:              float = 60.0
    elbow_bottom_min:           float = 160.0
    elbow_bottom_max:           float = 180.0
    shoulder_retract_threshold: float = 50.0
    swing_tolerance_deg:        float = 15.0
    weights: Dict[str, float] = field(default_factory=lambda: {
        "elbow_rom":        35.0,
        "shoulder_retract": 25.0,
        "no_swing":         20.0,
        "full_extension":   20.0,
    })


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class JumpingJackThresholds:
    """تمرين القفز المتفرج / Jumping Jack"""
    # زاوية الذراعين في الأعلى (الكتف - الورك - الرسغ)
    arms_up_min:     float = 150.0   # الذراعان مرفوعتان
    arms_down_max:   float = 40.0    # الذراعان منخفضتان
    # زاوية الساقين (عرض القدمين نسبةً لعرض الكتفين)
    legs_wide_ratio: float = 1.4     # نسبة عرض القدمين / عرض الكتفين
    legs_close_ratio:float = 0.6
    # الحد الأدنى لزاوية الورك عند الفتح
    hip_open_min:    float = 30.0
    weights: Dict[str, float] = field(default_factory=lambda: {
        "arms_rom":   35.0,
        "legs_rom":   35.0,
        "sync":       20.0,
        "landing":    10.0,
    })


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CrunchThresholds:
    """تمرين البطن (Crunch / Situp)"""
    # زاوية الورك عند أعلى نقطة (الظهر يرتفع عن الأرض)
    hip_up_max:      float = 80.0    # < 80° = وصل الأعلى
    hip_down_min:    float = 160.0   # > 160° = مستلقٍ بالكامل
    # زاوية الرقبة (لا تسحب الرقبة)
    neck_pull_max:   float = 120.0   # إذا أقل من 120 = يسحب رقبته
    weights: Dict[str, float] = field(default_factory=lambda: {
        "rom":        40.0,
        "neck_safe":  35.0,
        "controlled": 25.0,
    })


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SquatThresholds:
    """تمرين الجلسة (Squat)"""
    # زاوية الركبة
    knee_down_min:   float = 70.0    # أدنى نقطة
    knee_down_max:   float = 100.0
    knee_up_min:     float = 160.0   # واقف مكتمل
    knee_up_max:     float = 180.0
    # استقامة الظهر (الورك - الكتف - الرأس)
    back_lean_max:   float = 40.0    # الحد الأقصى للانحناء للأمام
    # الركبتان تنهاران للداخل (knee valgus)
    knee_valgus_threshold: float = 160.0
    weights: Dict[str, float] = field(default_factory=lambda: {
        "knee_rom":    35.0,
        "back_angle":  30.0,
        "knee_track":  20.0,
        "heel_down":   15.0,
    })


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RepCounterConfig:
    """إعدادات عداد التكرارات / Rep counter config"""
    confirmation_frames: int   = 4
    min_angle_delta:     float = 40.0
    min_rep_interval:    float = 0.8


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class VisualizationConfig:
    """إعدادات الرسم / Drawing config"""
    skeleton_color_good:  Tuple[int, int, int] = (0, 220, 0)
    skeleton_color_warn:  Tuple[int, int, int] = (0, 165, 255)
    skeleton_color_bad:   Tuple[int, int, int] = (0, 50, 255)
    landmark_color:       Tuple[int, int, int] = (255, 255, 255)
    angle_text_color:     Tuple[int, int, int] = (255, 255, 0)
    line_thickness:       int   = 3
    landmark_radius:      int   = 6
    thick_line:           int   = 5
    font_size_large:      int   = 28
    font_size_medium:     int   = 22
    font_size_small:      int   = 17
    overlay_alpha:        float = 0.6


# ─────────────────────────────────────────────────────────────────────────────
# Singleton instances
# ─────────────────────────────────────────────────────────────────────────────
PUSHUP_CFG       = PushupThresholds()
PULLUP_CFG       = PullupThresholds()
JUMPJACK_CFG     = JumpingJackThresholds()
CRUNCH_CFG       = CrunchThresholds()
SQUAT_CFG        = SquatThresholds()
REP_CFG          = RepCounterConfig()
VIZ_CFG          = VisualizationConfig()
