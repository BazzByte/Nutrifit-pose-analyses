# =============================================================================
# visualization/arabic_renderer.py
# محرك رسم النصوص العربية — كامل ومُعتمد على Pillow
# Arabic text renderer — production-ready, Pillow-based
#
# يحل المشكلة الأساسية: OpenCV لا يدعم:
#   ✗ العربية (تظهر ???? أو مقلوبة)
#   ✗ الترتيب RTL (يمين لشمال)
#   ✗ الحروف المتصلة (ك + ت + ب = كتب)
#
# الحل بثلاث طبقات:
#   1. arabic_reshaper  → يصل الحروف (ك+ت+ب = كتب)
#   2. python-bidi      → يرتب RTL صحيح
#   3. Pillow           → يرسم على الصورة بخط TrueType
#   Fallback           → OpenCV عادي (إنجليزي فقط)
# =============================================================================

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# ─── Pillow ─────────────────────────────────────────────────────────────────
try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False
    print("[WARNING] Pillow not installed. Run: pip install Pillow", file=sys.stderr)

# ─── arabic-reshaper ─────────────────────────────────────────────────────────
try:
    import arabic_reshaper
    _HAS_RESHAPER = True
except ImportError:
    _HAS_RESHAPER = False

# ─── python-bidi ─────────────────────────────────────────────────────────────
try:
    from bidi.algorithm import get_display
    _HAS_BIDI = True
except ImportError:
    _HAS_BIDI = False


# =============================================================================
# Font resolution
# =============================================================================
_ASSETS_DIR = Path(__file__).parent.parent / "assets" / "fonts"
_FONT_FILE  = _ASSETS_DIR / "NotoSansArabic-Regular.ttf"

_SYSTEM_FONTS = [
    Path("/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("C:/Windows/Fonts/tahoma.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
]

_NOTO_URL = (
    "https://github.com/googlefonts/noto-fonts/raw/main/"
    "hinted/ttf/NotoSansArabic/NotoSansArabic-Regular.ttf"
)


def _resolve_font() -> Optional[Path]:
    """Find font: local assets → system → auto-download."""
    if _FONT_FILE.exists():
        return _FONT_FILE
    for p in _SYSTEM_FONTS:
        if p.exists():
            return p
    try:
        _ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Downloading Arabic font to {_FONT_FILE} ...")
        urllib.request.urlretrieve(_NOTO_URL, _FONT_FILE)
        print("[INFO] Font downloaded successfully.")
        return _FONT_FILE
    except Exception as e:
        print(f"[WARNING] Font download failed: {e}", file=sys.stderr)
        return None


_FONT_PATH: Optional[Path] = _resolve_font() if _HAS_PIL else None


# =============================================================================
def _prepare_arabic(text: str) -> str:
    """
    أعد تشكيل النص العربي وطبّق bidi.
    Reshape Arabic and apply bidi algorithm for correct RTL rendering.
    """
    if not text:
        return text
    if _HAS_RESHAPER:
        try:
            text = arabic_reshaper.reshape(text)
        except Exception:
            pass
    if _HAS_BIDI:
        try:
            text = get_display(text)
        except Exception:
            pass
    return text


# =============================================================================
class TextRenderer:
    """
    محرك رسم النصوص يدعم العربية والإنجليزية.
    Text renderer supporting Arabic (RTL) and English (LTR).

    Usage:
        renderer = TextRenderer()
        frame = renderer.put_text(frame, "مرحبا بك", (10, 50), font_size=26)
        frame = renderer.put_text(frame, "Hello",    (10, 80), font_size=26)
    """

    def __init__(self) -> None:
        self._font_cache: dict[int, object] = {}
        self._arabic_ok  = _HAS_PIL and _FONT_PATH is not None
        self._pil_ok     = _HAS_PIL

    # ──────────────────────────────────────────────────────────────────── #
    def put_text(
        self,
        frame:     np.ndarray,
        text:      str,
        pos:       Tuple[int, int],
        font_size: int = 22,
        color:     Tuple[int, int, int] = (255, 255, 255),
        bg_color:  Optional[Tuple[int, int, int]] = None,
        bg_alpha:  float = 0.60,
        bold:      bool = False,
    ) -> np.ndarray:
        """
        ارسم نصاً على frame.
        Draw text on a BGR frame. Returns modified frame.
        """
        if not text or frame is None:
            return frame
        if self._arabic_ok:
            return self._pil_draw(frame, text, pos, font_size,
                                  color, bg_color, bg_alpha, bold)
        return self._cv2_draw(frame, text, pos, font_size, color)

    # ──────────────────────────────────────────────────────────────────── #
    def put_text_centered(
        self,
        frame:     np.ndarray,
        text:      str,
        y:         int,
        font_size: int = 22,
        color:     Tuple[int, int, int] = (255, 255, 255),
        bg_color:  Optional[Tuple[int, int, int]] = None,
        bg_alpha:  float = 0.60,
        bold:      bool = False,
    ) -> np.ndarray:
        """ارسم النص في المنتصف الأفقي / Draw text horizontally centered."""
        w = frame.shape[1]
        tw, _ = self.get_text_size(text, font_size)
        x = max(4, (w - tw) // 2)
        return self.put_text(frame, text, (x, y), font_size,
                             color, bg_color, bg_alpha, bold)

    # ──────────────────────────────────────────────────────────────────── #
    def get_text_size(self, text: str, font_size: int = 22) -> Tuple[int, int]:
        """احسب (عرض، ارتفاع) النص / Get (width, height) of text."""
        if self._pil_ok and _FONT_PATH:
            try:
                font  = self._get_font(font_size)
                prep  = _prepare_arabic(text)
                dummy = Image.new("RGB", (1, 1))
                draw  = ImageDraw.Draw(dummy)
                bbox  = draw.textbbox((0, 0), prep, font=font)
                return (max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1]))
            except Exception:
                pass
        scale = font_size / 30.0
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        return (tw, th)

    # ──────────────────────────────────────────────────────────────────── #
    def _get_font(self, size: int) -> "ImageFont.FreeTypeFont":
        if size not in self._font_cache:
            try:
                self._font_cache[size] = ImageFont.truetype(str(_FONT_PATH), size)
            except Exception:
                self._font_cache[size] = ImageFont.load_default()
        return self._font_cache[size]

    def _pil_draw(self, frame, text, pos, size, color, bg_color, bg_alpha, bold):
        try:
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil  = Image.fromarray(rgb)
            draw = ImageDraw.Draw(pil, "RGBA")
            font = self._get_font(size)
            prep = _prepare_arabic(text)
            rgb_c = (color[2], color[1], color[0])   # BGR→RGB

            if bg_color is not None:
                bbox = draw.textbbox(pos, prep, font=font)
                pad  = 6
                rgba = (bg_color[2], bg_color[1], bg_color[0], int(255 * bg_alpha))
                draw.rectangle(
                    (bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad),
                    fill=rgba,
                )

            draw.text(pos, prep, font=font, fill=rgb_c)
            if bold:
                draw.text((pos[0]+1, pos[1]),   prep, font=font, fill=rgb_c)
                draw.text((pos[0],   pos[1]+1), prep, font=font, fill=rgb_c)

            return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            return self._cv2_draw(frame, text, pos, size, color)

    def _cv2_draw(self, frame, text, pos, size, color):
        """Fallback OpenCV renderer — English only."""
        scale = size / 30.0
        cv2.putText(frame, text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        return frame

    @property
    def arabic_supported(self) -> bool:
        return self._arabic_ok and _HAS_RESHAPER and _HAS_BIDI

    @property
    def status(self) -> str:
        if self.arabic_supported:
            return "Full Arabic support (Pillow + reshaper + bidi)"
        missing = []
        if not _HAS_PIL:      missing.append("Pillow")
        if not _HAS_RESHAPER: missing.append("arabic-reshaper")
        if not _HAS_BIDI:     missing.append("python-bidi")
        if _FONT_PATH is None: missing.append("Arabic font")
        return f"Partial — missing: {', '.join(missing)}"


# Singleton
renderer = TextRenderer()
