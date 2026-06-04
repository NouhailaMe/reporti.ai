@echo off
title 🚀 AI Database Assistant Launcher
echo Starting Ollama...
start /B ollama serve
timeout /t 5 /nobreak
echo Starting AI Assistant...
cd /d "%~dp0"
if exist "..\venv\Scripts\activate.bat" call ..\venv\Scripts\activate.bat
python main.py
pause
