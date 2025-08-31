@echo off
echo Starting RoBERTa Backend Service...
echo ====================================

cd /d "backend"

echo Starting RoBERTa Backend on port 8002...
python app.py

pause
