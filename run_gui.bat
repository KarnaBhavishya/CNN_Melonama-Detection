@echo off
title Melanoma Detection Desktop GUI
echo ===================================================
echo Starting Melanoma Detection Desktop Application...
echo ===================================================
cd /d "%~dp0Final Year Project Batch-14(CNN-Based Melanoma Detection from Skin Lesion Images)"
"..\env\Scripts\python.exe" predict.py
pause
