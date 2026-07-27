@echo off
chcp 65001 > nul
title Qari Style Profile Builder
cd /d "%~dp0"
echo ============================================================
echo   Building Qari Recitation Style Profiles
echo ============================================================
echo.
set PYTHONIOENCODING=utf-8
venv\Scripts\python.exe matching\build_style_profiles.py
echo.
if %ERRORLEVEL% EQU 0 (
    echo SUCCESS - Profiles saved to matching\style_profiles.pkl
    echo You can now start the server with: python api\main.py
) else (
    echo ERROR - Profile build failed. Check output above.
)
echo.
pause
