@echo off
title Melanoma Detection Web Application
echo ===================================================
echo Starting Melanoma Detection Web Server (Flask)
echo Opening http://127.0.0.1:5000 in your browser...
echo ===================================================
cd /d "%~dp0"
start http://127.0.0.1:5000/
"..\env\Scripts\python.exe" app.py
pause
