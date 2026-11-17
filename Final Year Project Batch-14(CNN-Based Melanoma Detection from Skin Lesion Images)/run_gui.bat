@echo off
title Melanoma Detection Desktop GUI
echo ===================================================
echo Starting Melanoma Detection Desktop Application...
echo ===================================================
cd /d "%~dp0"
"..\env\Scripts\python.exe" predict.py
pause
