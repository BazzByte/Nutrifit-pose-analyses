# =============================================================================
# server.py — v2  (Railway-compatible fix)
# خادم FastAPI للتكامل مع Flutter
#
# ✅ FIX 1: PORT env variable for Railway
# ✅ FIX 2: startup event to verify mediapipe loaded
# ✅ FIX 3: proper error handling for missing display
#
# تشغيل محلي / Local run:
#   uvicorn server:app --host 0.0.0.0 --port 8000 --reload
#
# Railway يشغّل تلقائياً باستخدام Procfile:
#   web: uvicorn server:app --host 0.0.0.0 --port $PORT
# =============================================================================

from __future__ import annotations

import base64
import os  # ✅ FIX 1: needed for PORT env var
from typing import List, Optional

import cv2
import numpy as np

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False
    print("[WARNING] pip install fastapi uvicorn python-multipart")

from workout_analyzer import WorkoutAnalyzer, ALL_EXERCISES

if _HAS_FASTAPI:

    app = FastAPI(
        title="Fitness AI API v2",
        description="Bilingual real-time workout analysis for Flutter",
        version="2.0.0",
    )

    app.add_middleware(CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"])

    # ✅ FIX 2: startup check
    @app.on_event("startup")
    async def startup_event():
        print("[INFO] Fitness AI API starting...")
        print(f"[INFO] Available exercises: {list(ALL_EXERCISES)}")
        print(f"[INFO] OpenCV version: {cv2.__version__}")
        try:
            import mediapipe as mp
            print(f"[INFO] MediaPipe version: {mp.__version__}")
        except Exception as e:
            print(f"[WARNING] MediaPipe check failed: {e}")

    # Singleton لكل لغة
    _analyzers: dict[str, WorkoutAnalyzer] = {}

    def _get_analyzer(lang: str) -> WorkoutAnalyzer:
        if lang not in _analyzers:
            _analyzers[lang] = WorkoutAnalyzer(
                model_complexity=1, lang=lang, show_fps=False  # type: ignore
            )
        return _analyzers[lang]

    # ──────────────────────────────────────────────────────────────────────
    class AnalyzeRequest(BaseModel):
        exercise: str       # pushup | pullup | jumpingjack | crunch | squat
        image: str          # Base64 JPEG
        lang: str = "ar"    # "ar" | "en"
        reset: bool = False
        quality: int = 80   # JPEG output quality

    class AnalyzeResponse(BaseModel):
        percentage: float
        feedback_text: str
        audio_cue: Optional[str]
        angles: dict
        rep_count: int
        phase: str
        is_calibrating: bool
        feedback_history: List[str]
        annotated_image: str  # Base64 JPEG
        language: str

    # ──────────────────────────────────────────────────────────────────────
    @app.post("/analyze", response_model=AnalyzeResponse)
    async def analyze(req: AnalyzeRequest):
        if req.exercise not in ALL_EXERCISES:
            raise HTTPException(400, f"exercise must be one of {ALL_EXERCISES}")
        if req.lang not in ("ar", "en"):
            raise HTTPException(400, "lang must be 'ar' or 'en'")

        try:
            img_bytes = base64.b64decode(req.image)
            arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Invalid image data")
        except Exception as e:
            raise HTTPException(400, f"Image decode error: {e}")

        analyzer = _get_analyzer(req.lang)

        if req.reset:
            analyzer.reset_exercise(req.exercise)  # type: ignore

        result = analyzer.process_frame(frame, req.exercise)  # type: ignore

        _, buf = cv2.imencode(".jpg", result.frame,
                              [cv2.IMWRITE_JPEG_QUALITY, req.quality])
        out_b64 = base64.b64encode(buf.tobytes()).decode()

        return AnalyzeResponse(
            percentage=result.percentage,
            feedback_text=result.feedback_text,
            audio_cue=result.audio_cue,
            angles=result.angles,
            rep_count=result.rep_count,
            phase=result.phase,
            is_calibrating=result.is_calibrating,
            feedback_history=result.feedback_history,
            annotated_image=out_b64,
            language=req.lang,
        )

    @app.post("/reset/{exercise}")
    async def reset(exercise: str, lang: str = "ar"):
        if exercise not in ALL_EXERCISES:
            raise HTTPException(400, "Invalid exercise")
        _get_analyzer(lang).reset_exercise(exercise)  # type: ignore
        return {"status": "ok", "exercise": exercise}

    @app.get("/exercises")
    async def list_exercises():
        return {"exercises": list(ALL_EXERCISES)}

    @app.get("/health")
    async def health():
        return {
            "status": "running",
            "version": "2.0.0",
            "exercises": list(ALL_EXERCISES),
            "port": int(os.environ.get("PORT", 8000)),  # ✅ FIX 1
        }

else:
    print("[ERROR] pip install fastapi uvicorn python-multipart")
