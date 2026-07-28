"""
Qari Voice Recognition — FastAPI Microservice
==============================================
Pure backend service. No frontend. No static files.

Single feature: identify which of the 12 Qaris an uploaded recitation
most closely resembles, using ECAPA-TDNN speaker embeddings.

Endpoints
---------
POST /identify-qari   — upload audio, receive top-5 Qari matches (primary)
GET  /health          — liveness + readiness check
GET  /                — service info (Swagger link included)

Swagger UI : http://localhost:8000/docs
ReDoc      : http://localhost:8000/redoc
"""

import os
import sys
import logging
import tempfile
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Project root on sys.path so all internal imports resolve
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from api.audio_validator import validate_audio_file, ValidationResult
from matching.similarity import (
    get_top5_similar,
    _load_classifier,
    _load_reference_embeddings,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
(BASE_DIR / "logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR / "logs" / "api.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("qari_api")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Qari Voice Recognition API",
    description=(
        "Upload a Quranic recitation audio clip and receive the **top-5 Qaris** "
        "whose voice most closely matches yours.\n\n"
        "Matching is performed via **ECAPA-TDNN** speaker embeddings "
        "(192-dimensional cosine similarity against pre-built Qari centroids).\n\n"
        "Accuracy: **99.73 % top-1** on the 12-Qari benchmark dataset.\n\n"
        "**Recommended input:** 10 s or more of clean recitation (2-3 Ayahs is enough)."
    ),
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow all origins so any external frontend can call this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Startup — pre-load model so first request is not slow
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Warming up ECAPA-TDNN model...")
        _load_classifier()
        refs = _load_reference_embeddings()
        logger.info(
            f"Ready. {len(refs)} Qari centroids loaded: {sorted(refs.keys())}"
        )
    except FileNotFoundError:
        logger.warning(
            "Reference embeddings not found. "
            "Run: python matching/build_reference_embeddings.py  then restart."
        )
    except Exception as exc:
        logger.error(f"Startup error: {exc}")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}


def _check_extension(filename: str) -> None:
    ext = Path(filename or "upload.wav").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Accepted: wav, mp3, m4a, flac.",
        )


def _confidence(pct: float) -> str:
    if pct >= 65:
        return "high"
    if pct >= 35:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------
@app.get("/", tags=["Info"], summary="Service info")
async def root():
    """Returns service name, version and the Swagger UI link."""
    return {
        "service":  "Qari Voice Recognition API",
        "version":  "4.0.0",
        "engine":   "ECAPA-TDNN speaker embeddings",
        "status":   "running",
        "swagger":  "/docs",
        "endpoint": "POST /identify-qari",
    }


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Info"], summary="Health / readiness check")
async def health():
    """
    Returns **healthy** when the ECAPA model and reference embeddings are loaded.
    Returns **degraded** when they are missing (service started before data was built).
    """
    try:
        refs = _load_reference_embeddings()
        return {
            "status":     "healthy",
            "engine":     "ECAPA-TDNN",
            "num_qaris":  len(refs),
            "qaris":      sorted(refs.keys()),
        }
    except FileNotFoundError:
        return JSONResponse(
            status_code=503,
            content={
                "status":  "degraded",
                "reason":  "Reference embeddings not found.",
                "fix":     "Run: python matching/build_reference_embeddings.py  then restart.",
            },
        )
    except Exception as exc:
        logger.error(f"Health check error: {exc}")
        raise HTTPException(status_code=503, detail="Service unavailable")


# ---------------------------------------------------------------------------
# POST /identify-qari  — the only feature endpoint
# ---------------------------------------------------------------------------
@app.post(
    "/identify-qari",
    tags=["Identification"],
    summary="Identify top-5 matching Qaris from a recitation audio clip",
    response_description="Top-5 Qari matches with similarity scores",
)
async def identify_qari(
    audio_file: UploadFile = File(
        ...,
        description=(
            "Audio file of the recitation. "
            "Accepted formats: WAV, MP3, M4A, FLAC. Max size: 50 MB. "
            "Recommended: 10 s or more of clean recitation."
        ),
    ),
):
    """
    Upload a Quranic recitation clip and get the **top-5 Qaris** whose voice
    most closely matches the uploaded audio.

    ### How it works
    1. Audio is resampled to 16 kHz mono and peak-normalised.
    2. A 4-second sliding window (2-second hop) is applied across the clip.
    3. Each window is passed through **ECAPA-TDNN** → 192-dim embedding.
    4. All window embeddings are averaged and L2-normalised.
    5. Cosine similarity is computed against 12 pre-built Qari centroids.
    6. Results are mapped to percentages and the top-5 are returned.

    ### Response fields
    | Field | Type | Description |
    |---|---|---|
    | `success` | bool | Always `true` on a 200 response |
    | `top_match` | object | The single best-matching Qari |
    | `top_5` | array | Ranked list of the 5 best matches |
    | `interpretation` | string | Human-readable summary sentence |

    ### Each match object
    | Field | Type | Description |
    |---|---|---|
    | `rank` | int | 1 = best match |
    | `qari` | string | Qari name |
    | `similarity_percent` | float | 0–100 % similarity score |
    | `raw_cosine_score` | float | Raw cosine similarity (0–1) |
    | `confidence` | string | `"high"` / `"medium"` / `"low"` |
    """
    temp_path = None
    try:
        # 1. Extension check
        _check_extension(audio_file.filename or "")

        # 2. Save to temp file
        suffix = Path(audio_file.filename or "upload.wav").suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name
            tmp.write(await audio_file.read())

        # 3. Audio content validation (duration, silence, sample rate)
        validation = validate_audio_file(temp_path)
        if not validation.is_valid:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Audio validation failed.",
                    "errors":  validation.errors,
                },
            )

        logger.info(
            f"[identify-qari] file={audio_file.filename}  "
            f"duration={validation.audio_info.get('duration_sec')}s  "
            f"sr={validation.audio_info.get('sample_rate')}Hz"
        )

        # 4. ECAPA-TDNN: rank all 12 Qaris, keep top 5
        ranked = get_top5_similar(temp_path, top_n=5)

        # 5. Build clean response
        top_5 = [
            {
                "rank":               i + 1,
                "qari":               r["qari_id"],
                "similarity_percent": r["similarity_percent"],
                "raw_cosine_score":   r["similarity_score"],
                "confidence":         _confidence(r["similarity_percent"]),
            }
            for i, r in enumerate(ranked)
        ]

        best = top_5[0]

        return {
            "success": True,
            "top_match": {
                "qari":               best["qari"],
                "similarity_percent": best["similarity_percent"],
                "confidence":         best["confidence"],
            },
            "top_5": top_5,
            "interpretation": (
                f"Your recitation most closely resembles {best['qari']} "
                f"({best['similarity_percent']}% similarity, "
                f"{best['confidence']} confidence)."
            ),
            "audio_info": {
                "filename":     audio_file.filename,
                "duration_sec": validation.audio_info.get("duration_sec"),
                "sample_rate":  validation.audio_info.get("sample_rate"),
                "file_size_mb": validation.audio_info.get("file_size_mb"),
            },
            "warnings": validation.warnings,
        }

    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=(
                "Reference embeddings not built. "
                "Run: python matching/build_reference_embeddings.py  then restart."
            ),
        )
    except Exception as exc:
        logger.error(f"[identify-qari] unhandled error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


# ---------------------------------------------------------------------------
# Generic error handlers
# ---------------------------------------------------------------------------
@app.exception_handler(404)
async def not_found(_req, _exc):
    return JSONResponse(
        status_code=404,
        content={"success": False, "error": "Endpoint not found. See /docs for available endpoints."},
    )


@app.exception_handler(500)
async def server_error(_req, exc):
    logger.error(f"Unhandled 500: {exc}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error."},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )