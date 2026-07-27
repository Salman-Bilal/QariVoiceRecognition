@echo off
REM Qari Voice Recognition System - Windows Launcher
REM Double-click this file to start the application

echo ============================================================
echo  Qari Voice Recognition System - Starting...
echo ============================================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found. Creating one...
    python -m venv venv
    echo.
    echo Installing dependencies...
    call venv\Scripts\activate.bat
    pip install --upgrade pip
    pip install -r requirements.txt
) else (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo.
echo Starting the application...
echo.
python app.py

pause
