#!/bin/bash
# Qari Voice Recognition System - Linux/Mac Launcher

echo "============================================================"
echo " Qari Voice Recognition System - Starting..."
echo "============================================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo ""
    echo "Installing dependencies..."
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo ""
echo "Starting the application..."
echo ""
python app.py
