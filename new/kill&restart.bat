@echo off
echo 🔪 Killing old processes...
taskkill /F /IM python.exe /T 2>nul
timeout /t 2 /nobreak >nul
echo 🚀 Starting AI Assistant...
cd /d "C:\Users\Nouhaila\Desktop\reporti.ai\new"
call ..\venv\Scripts\activate.bat
python main.py
pause
