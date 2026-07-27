#!/usr/bin/env python3
"""
Qari Voice Recognition System - Main Entry Point
Starts the FastAPI backend server
"""

import os
import sys
from pathlib import Path
import webbrowser
import time
import uvicorn

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

def main():
    """Main entry point for the application"""
    print("=" * 70)
    print(" 🎙️  QARI VOICE RECOGNITION SYSTEM")
    print("=" * 70)
    print()
    print("Starting the API server...")
    print()
    
    # Create logs directory if it doesn't exist
    logs_dir = BASE_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Server configuration
    host = "0.0.0.0"
    port = 8000
    
    print(f"✅ Server will start at: http://localhost:{port}")
    print(f"✅ Frontend interface: http://localhost:{port}/static/index.html")
    print(f"✅ API documentation: http://localhost:{port}/docs")
    print()
    print("Press CTRL+C to stop the server")
    print("=" * 70)
    print()
    
    # Auto-open browser after a short delay
    def open_browser():
        time.sleep(2)
        try:
            webbrowser.open(f"http://localhost:{port}/static/index.html")
        except:
            pass
    
    import threading
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Start the server
    try:
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            reload=False,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
