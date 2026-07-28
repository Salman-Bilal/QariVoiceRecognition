"""
FastAPI Backend for Qari Voice Recognition System
Provides endpoints for audio upload, analysis, and Qari comparison

Matching engine: style-based (pitch contour + rhythm + breath pattern)
  /api/identify-qari      — ranks all Qaris by recitation style similarity
  /api/compare-all-qaris  — full breakdown of style similarity vs all Qaris
  /api/analyze-recitation — detailed timing/melody/breath score vs one Qari (unchanged)
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict
import tempfile
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Import audio validator
from api.audio_validator import validate_audio_file as external_validate_audio, ValidationResult

# ── Style-based matching engine (replaces ECAPA for identification) ──────────
from matching.style_similarity import compare_style_to_all_qaris, _get_profiles_and_stats
from matching.style_profiles import load_style_profiles, STYLE_PROFILES_PATH

# ── Recitation quality analysis (unchanged — used by /api/analyze-recitation) ─
from analysis.aggregate import generate_recitation_report
from analysis.config import REFERENCE_QARI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(BASE_DIR / 'logs' / 'api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Qari Voice Recognition API",
    description=(
        "AI-powered Quranic recitation analysis and Qari identification system.\n\n"
        "Matching is based on recitation STYLE (pitch contour, rhythm, breath pattern), "
        "not speaker identity."
    ),
    version="2.0.0"
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (frontend)
frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.on_event("startup")
async def startup_event():
    """Load style profiles on startup."""
    try:
        logger.info("Loading Qari style profiles...")
        profiles, _ = _get_profiles_and_stats()
        logger.info(f"✅ Loaded style profiles for {len(profiles)} Qaris: {sorted(profiles.keys())}")
    except FileNotFoundError:
        logger.warning(
            "⚠️  Style profiles not found. "
            "Run: python matching/build_style_profiles.py — then restart the server."
        )
    except Exception as e:
        logger.error(f"Failed to load style profiles: {e}")


def validate_upload_metadata(file: UploadFile):
    """Basic validation for uploaded file metadata (format & size)."""
    allowed_extensions = ['.wav', '.mp3', '.m4a', '.flac']
    filename = getattr(file, 'filename', None) or str(file)
    file_ext = Path(filename).suffix.lower()

    if file_ext not in allowed_extensions:
        return False, f"Invalid file format. Allowed: {', '.join(allowed_extensions)}"

    if hasattr(file, 'size') and file.size and file.size > 50 * 1024 * 1024:
        return False, "File too large. Maximum size: 50MB"

    return True, None


# ─────────────────────────────────────────────────────────────────────────────
# Info endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "Qari Voice Recognition API",
        "version": "2.0.0",
        "matching_engine": "Style-based (pitch + rhythm + breath)",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "identify_qari": "/api/identify-qari",
            "analyze_recitation": "/api/analyze-recitation",
            "compare_all_qaris": "/api/compare-all-qaris",
            "list_qaris": "/api/list-qaris",
            "available_surahs": "/api/available-surahs",
        }
    }


@app.get("/health")
async def health_check():
    try:
        profiles, _ = _get_profiles_and_stats()
        return {
            "status": "healthy",
            "style_profiles_loaded": True,
            "num_qaris": len(profiles),
            "qaris": sorted(profiles.keys()),
        }
    except FileNotFoundError:
        return {
            "status": "degraded",
            "style_profiles_loaded": False,
            "message": "Run python matching/build_style_profiles.py then restart.",
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.get("/api/list-qaris")
async def list_qaris():
    """Get list of available Qaris."""
    try:
        profiles, _ = _get_profiles_and_stats()
        qaris = sorted(profiles.keys())
        return {
            "success": True,
            "qaris": qaris,
            "count": len(qaris),
            "default_reference": REFERENCE_QARI,
        }
    except Exception as e:
        logger.error(f"Failed to list Qaris: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/available-surahs")
async def available_surahs():
    """Get list of available Surahs for analysis."""
    normalized_dir = BASE_DIR / "dataset" / "processed" / "normalized"

    if not normalized_dir.exists():
        return {"success": False, "error": "Reference dataset not found"}

    qari_dirs = [d for d in normalized_dir.iterdir() if d.is_dir()]
    if not qari_dirs:
        return {"success": False, "error": "No Qari directories found"}

    first_qari_dir = qari_dirs[0]
    surah_files = [f.stem for f in first_qari_dir.glob("*.wav")]

    return {
        "success": True,
        "surahs": sorted(surah_files),
        "count": len(surah_files)
    }


# ─────────────────────────────────────────────────────────────────────────────
# /api/identify-qari — style-based ranking
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/identify-qari")
async def identify_qari(
    audio_file: UploadFile = File(...),
    top_k: int = Form(5)
):
    """
    Identify which Qari's recitation style the uploaded audio most resembles.

    Scores are based on three style dimensions:
      - Pitch similarity  (40%) — melody contour, vocal range, voiced fraction
      - Rhythm similarity (35%) — tempo, syllable rate, rhythmic regularity
      - Breath similarity (25%) — phrasing density, pause pattern

    Returns ranked list of similar Qaris with per-dimension breakdown.
    """
    temp_path = None

    try:
        # 1. Validate file metadata
        is_valid, error_msg = validate_upload_metadata(audio_file)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        # 2. Save to temp file
        suffix = Path(audio_file.filename or "temp.wav").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name
            content = await audio_file.read()
            tmp.write(content)

        # 3. Validate audio content
        validation = external_validate_audio(temp_path)
        if not validation.is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Audio validation failed: {'; '.join(validation.errors)}"
            )
        for w in validation.warnings:
            logger.warning(f"Audio warning: {w}")

        logger.info(f"[identify-qari] {audio_file.filename} | {validation.audio_info}")

        # 4. Style-based comparison
        all_results = compare_style_to_all_qaris(temp_path)

        # 5. Build response
        all_matches = [
            {
                "qari":          r["qari"],
                "similarity":    r["overall_score"],
                "pitch_score":   r["pitch_score"],
                "rhythm_score":  r["rhythm_score"],
                "breath_score":  r["breath_score"],
                "match_level":   r["match_level"],
                "confidence": (
                    "high"   if r["overall_score"] > 65 else
                    "medium" if r["overall_score"] > 40 else
                    "low"
                ),
            }
            for r in all_results
        ]

        top_match = all_matches[0]
        top_k_matches = all_matches[:top_k]

        return {
            "success": True,
            "filename": audio_file.filename,
            "audio_info": validation.audio_info,
            "warnings": validation.warnings,
            "matching_engine": "style-based (pitch + rhythm + breath)",
            "top_match": top_match,
            "top_k_matches": top_k_matches,
            "all_matches": all_matches,
            "interpretation": (
                f"Your recitation style most closely resembles {top_match['qari']} "
                f"({top_match['similarity']}% overall style match)"
            ),
            "score_breakdown": {
                "pitch_weight":  "40% — melody contour and vocal range",
                "rhythm_weight": "35% — tempo and syllable timing",
                "breath_weight": "25% — phrasing and pause pattern",
            }
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"Style profiles missing: {e}")
        raise HTTPException(
            status_code=503,
            detail=(
                "Style profiles not built yet. "
                "Run: python matching/build_style_profiles.py — then restart the server."
            )
        )
    except Exception as e:
        logger.error(f"Error in identify_qari: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


# ─────────────────────────────────────────────────────────────────────────────
# /api/analyze-recitation — detailed quality analysis (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/analyze-recitation")
async def analyze_recitation(
    audio_file: UploadFile = File(...),
    surah_name: str = Form(...),
    reference_qari: Optional[str] = Form(None)
):
    """
    Analyze recitation quality against a specific reference Qari for a given Surah.
    Returns timing, melody, and breath scores with detailed feedback.

    This endpoint is for "how well did I recite THIS surah like THIS Qari?"
    For "which Qari do I sound most like overall?" use /api/identify-qari.
    """
    temp_path = None

    try:
        is_valid, error_msg = validate_upload_metadata(audio_file)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            temp_path = tmp.name
            content = await audio_file.read()
            tmp.write(content)

        validation = external_validate_audio(temp_path)
        if not validation.is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Audio validation failed: {'; '.join(validation.errors)}"
            )
        for w in validation.warnings:
            logger.warning(f"Audio warning: {w}")

        logger.info(
            f"[analyze-recitation] {surah_name} vs "
            f"{reference_qari or REFERENCE_QARI}"
        )

        report = generate_recitation_report(
            temp_path,
            surah_name,
            reference_qari=reference_qari
        )

        return {
            "success": True,
            "filename": audio_file.filename,
            "audio_info": validation.audio_info,
            "warnings": validation.warnings,
            "report": report
        }

    except FileNotFoundError as e:
        logger.error(f"Reference file not found: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Reference audio not found for '{surah_name}'. Check surah name format."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_recitation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


# ─────────────────────────────────────────────────────────────────────────────
# /api/compare-all-qaris — full style comparison
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/compare-all-qaris")
async def compare_all_qaris(audio_file: UploadFile = File(...)):
    """
    Compare user's recitation style against all available Qaris.

    Returns a comprehensive breakdown showing:
      - Overall style similarity % for each Qari
      - Individual pitch, rhythm, and breath scores
      - Match level label (Excellent / Good / Moderate / Weak)
      - Style description explaining which dimension was closest
    """
    temp_path = None

    try:
        is_valid, error_msg = validate_upload_metadata(audio_file)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        suffix = Path(audio_file.filename or "temp.wav").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name
            content = await audio_file.read()
            tmp.write(content)

        validation = external_validate_audio(temp_path)
        if not validation.is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Audio validation failed: {'; '.join(validation.errors)}"
            )
        for w in validation.warnings:
            logger.warning(f"Audio warning: {w}")

        logger.info(f"[compare-all-qaris] {audio_file.filename}")

        # Full style comparison
        all_results = compare_style_to_all_qaris(temp_path)

        comparisons = [
            {
                "qari":              r["qari"],
                "similarity_percent": r["overall_score"],
                "pitch_score":       r["pitch_score"],
                "rhythm_score":      r["rhythm_score"],
                "breath_score":      r["breath_score"],
                "match_level":       r["match_level"],
                "style_description": r["style_description"],
            }
            for r in all_results
        ]

        scores = [c["similarity_percent"] for c in comparisons]

        return {
            "success": True,
            "filename": audio_file.filename,
            "audio_info": validation.audio_info,
            "warnings": validation.warnings,
            "matching_engine": "style-based (pitch + rhythm + breath)",
            "total_qaris_compared": len(comparisons),
            "best_match": comparisons[0],
            "all_comparisons": comparisons,
            "statistics": {
                "highest_similarity": round(float(max(scores)), 2),
                "lowest_similarity":  round(float(min(scores)), 2),
                "average_similarity": round(float(np.mean(scores)), 2),
                "median_similarity":  round(float(np.median(scores)), 2),
            },
            "score_breakdown": {
                "pitch_weight":  "40% — melody contour and vocal range",
                "rhythm_weight": "35% — tempo and syllable timing",
                "breath_weight": "25% — phrasing and pause pattern",
            }
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"Style profiles missing: {e}")
        raise HTTPException(
            status_code=503,
            detail=(
                "Style profiles not built yet. "
                "Run: python matching/build_style_profiles.py — then restart the server."
            )
        )
    except Exception as e:
        logger.error(f"Error in compare_all_qaris: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Error handlers
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"success": False, "error": "Endpoint not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn

    (BASE_DIR / "logs").mkdir(exist_ok=True)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
