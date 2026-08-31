@echo off
title Industrial Steel Defect Inspection - Live Camera
cd /d "%~dp0"

echo ============================================================
echo   INDUSTRIAL STEEL DEFECT INSPECTION
echo   Live Camera Launcher
echo ============================================================
echo.

REM Activate the project's virtual environment.
if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
) else (
    echo ERROR: Virtual environment not found.
    echo Expected: %~dp0venv\Scripts\activate.bat
    echo.
    pause
    exit /b 1
)

REM Prefer the corrected live-camera application.
if exist "live_camera_v2.py" (
    set "APP=live_camera_v2.py"
) else if exist "live_camera.py" (
    set "APP=live_camera.py"
) else (
    echo ERROR: Live camera application not found.
    echo Expected live_camera_v2.py or live_camera.py
    echo.
    pause
    exit /b 1
)

echo Starting %APP%...
echo.
python "%APP%"

if errorlevel 1 (
    echo.
    echo ============================================================
    echo The application exited with an error.
    echo Check the message above for details.
    echo ============================================================
    pause
)
