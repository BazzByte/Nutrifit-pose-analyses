# =============================================================================
# visualization/i18n.py
# نظام الترجمة ثنائي اللغة (عربي / إنجليزي)
# Bilingual i18n system (Arabic / English)
# =============================================================================

from __future__ import annotations
from typing import Literal

Lang = Literal["ar", "en"]

# =============================================================================
# قاموس النصوص الكامل
# Full text dictionary
# =============================================================================
_TEXTS: dict[str, dict[Lang, str]] = {

    # ─── واجهة عامة / General UI ────────────────────────────────────────────
    "calibrating":          {"ar": "جارٍ المعايرة...",          "en": "Calibrating..."},
    "hold_start_pos":       {"ar": "ثبّت وضعيتك الابتدائية",    "en": "Hold starting position"},
    "no_person":            {"ar": "لا يوجد شخص في الإطار",     "en": "No person detected"},
    "check_camera":         {"ar": "تأكد من ظهور جسمك كاملاً",  "en": "Make sure your full body is visible"},
    "reps":                 {"ar": "تكرارات",                    "en": "Reps"},
    "form_score":           {"ar": "جودة الأداء",               "en": "Form Score"},
    "phase":                {"ar": "المرحلة",                   "en": "Phase"},
    "fps":                  {"ar": "إطار/ث",                    "en": "FPS"},
    "ready":                {"ar": "جاهز",                      "en": "Ready"},
    "going_down":           {"ar": "نزول ↓",                    "en": "Going Down ↓"},
    "bottom":               {"ar": "أدنى نقطة",                 "en": "Bottom"},
    "going_up":             {"ar": "صعود ↑",                    "en": "Going Up ↑"},
    "top":                  {"ar": "أعلى نقطة",                 "en": "Top"},
    "press_q":              {"ar": "Q للخروج | R للتعيين | T للتبديل", "en": "Q=Quit | R=Reset | T=Toggle"},

    # ─── Push-up feedback ───────────────────────────────────────────────────
    "pu_sagging":           {"ar": "لا تثنِ ظهرك — شدّ عضلات البطن",       "en": "Don't sag your back — tighten your core"},
    "pu_piking":            {"ar": "لا ترفع الورك — جسمك خط مستقيم",        "en": "Don't pike your hips — keep body straight"},
    "pu_head_down":         {"ar": "ارفع رأسك قليلاً — نظر للأمام",          "en": "Lift your head — look forward not down"},
    "pu_go_deeper":         {"ar": "انزل أكثر — الكوع 90° على الأقل",        "en": "Go deeper — elbow should reach 90°"},
    "pu_extend_arms":       {"ar": "امدد الذراعين بالكامل في الأعلى",         "en": "Fully extend arms at the top"},
    "pu_slow_descent":      {"ar": "أنزل ببطء (المرحلة السالبة مهمة)",        "en": "Slow down on the way down (negatives matter)"},
    "pu_elbows_close":      {"ar": "الكوعان قريبان جداً — افتح أكثر",          "en": "Elbows too close — flare them slightly"},
    "pu_excellent":         {"ar": "ممتاز! استمر بهذا الشكل 💪",              "en": "Excellent! Keep it up 💪"},
    "pu_good":              {"ar": "جيد — ركّز على إبقاء الجسم مستقيماً",     "en": "Good — focus on keeping body straight"},

    # ─── Pull-up feedback ───────────────────────────────────────────────────
    "plu_swinging":         {"ar": "توقف عن التأرجح — تحكم بالحركة",          "en": "Stop swinging — control your movement"},
    "plu_go_higher":        {"ar": "ارتفع أكثر — الذقن فوق العارضة",          "en": "Pull higher — chin over the bar"},
    "plu_full_hang":        {"ar": "امدد الذراعين بالكامل في الأسفل",          "en": "Fully extend arms at the bottom (dead hang)"},
    "plu_retract":          {"ar": "اشدد كتفيك للخلف ولأسفل عند الصعود",       "en": "Retract scapula — pull shoulders back & down"},
    "plu_reduce_swing":     {"ar": "قلل التأرجح — تحكم أكثر",                  "en": "Reduce swinging — more control"},
    "plu_slow_descent":     {"ar": "أنزل ببطء والتحكم في المرحلة السالبة",     "en": "Control the descent (negative phase)"},
    "plu_excellent":        {"ar": "ممتاز! أداء قوي ومتحكم 💪",               "en": "Excellent! Strong and controlled 💪"},
    "plu_good":             {"ar": "جيد — ركز على شد الكتفين في الأعلى",       "en": "Good — focus on shoulder retraction at top"},

    # ─── Jumping Jack feedback ───────────────────────────────────────────────
    "jj_arms_up":           {"ar": "ارفع الذراعين بالكامل فوق الرأس",         "en": "Raise arms fully above your head"},
    "jj_legs_wide":         {"ar": "افتح الساقين أكثر عند القفز",             "en": "Spread legs wider when jumping"},
    "jj_sync":              {"ar": "زامن حركة الذراعين والساقين معاً",         "en": "Sync arms and legs together"},
    "jj_land_soft":         {"ar": "انزل بلطف على أصابع القدم",               "en": "Land softly on the balls of your feet"},
    "jj_excellent":         {"ar": "ممتاز! إيقاع رائع 🌟",                    "en": "Excellent! Great rhythm 🌟"},
    "jj_good":              {"ar": "جيد — حافظ على الإيقاع المنتظم",           "en": "Good — maintain steady rhythm"},

    # ─── Crunch / Situp feedback ─────────────────────────────────────────────
    "cr_chin_chest":        {"ar": "لا تسحب رقبتك — الحركة من البطن",         "en": "Don't pull your neck — crunch from your core"},
    "cr_go_higher":         {"ar": "ارفع الكتفين أكثر عن الأرض",              "en": "Lift shoulders higher off the ground"},
    "cr_controlled":        {"ar": "أنزل ببطء — تحكم في الحركة",               "en": "Lower slowly — controlled movement"},
    "cr_breathe":           {"ar": "ازفر عند الرفع وشهّق عند النزول",          "en": "Exhale on the way up, inhale on the way down"},
    "cr_excellent":         {"ar": "ممتاز! عضلات بطن نارية 🔥",               "en": "Excellent! Core on fire 🔥"},
    "cr_good":              {"ar": "جيد — ركز على عضلات البطن لا الرقبة",      "en": "Good — focus on abs not your neck"},

    # ─── Squat feedback ─────────────────────────────────────────────────────
    "sq_deeper":            {"ar": "انزل أكثر — الفخذ موازٍ للأرض",           "en": "Go deeper — thighs parallel to ground"},
    "sq_knees_out":         {"ar": "الركبتان تنهاران للداخل — افتحهما",        "en": "Knees caving in — push them out"},
    "sq_back_straight":     {"ar": "لا تثنِ ظهرك للأمام — استقم",             "en": "Don't lean too far forward — stay upright"},
    "sq_heels_up":          {"ar": "لا ترفع كعبيك — اثبت على الأرض",          "en": "Don't lift heels — keep feet flat"},
    "sq_extend_top":        {"ar": "امدد الركبتين بالكامل في الأعلى",          "en": "Fully extend knees at the top"},
    "sq_excellent":         {"ar": "ممتاز! عمق مثالي 🏋️",                    "en": "Excellent! Perfect depth 🏋️"},
    "sq_good":              {"ar": "جيد — ركز على العمق والظهر المستقيم",      "en": "Good — focus on depth and straight back"},
}


# =============================================================================
class I18n:
    """
    مدير الترجمة — يُنشأ مرة واحدة ويُستخدم في كل مكان.
    Translation manager — instantiate once, use everywhere.
    """

    def __init__(self, lang: Lang = "ar") -> None:
        self.lang: Lang = lang

    def t(self, key: str, fallback: str = "") -> str:
        """
        ترجم مفتاحاً إلى اللغة الحالية.
        Translate a key to the current language.

        Args:
            key:      مفتاح النص في القاموس
            fallback: نص افتراضي إذا لم يوجد المفتاح

        Returns:
            النص المترجم أو الـ fallback
        """
        entry = _TEXTS.get(key)
        if entry is None:
            return fallback or key
        return entry.get(self.lang, entry.get("en", fallback or key))

    def set_lang(self, lang: Lang) -> None:
        """غيّر اللغة في أي وقت / Switch language at runtime."""
        self.lang = lang

    @property
    def is_arabic(self) -> bool:
        return self.lang == "ar"


# Singleton — استورده في أي مكان
# Import this singleton anywhere
i18n = I18n(lang="ar")
