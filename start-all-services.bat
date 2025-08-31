@echo off
echo Starting Complete Hate Speech Detection System
echo ================================================
echo.

echo This will start all services:
echo 1. AI-Backend (Primary - OpenAI + LangChain + RAG) - Port 8001
echo 2. RoBERTa Backend (Secondary/Fallback) - Port 8002  
echo 3. Frontend (Web Interface)
echo.

echo Starting AI-Backend Service...
start "AI-Backend" cmd /c "start-ai-backend.bat"

timeout /t 3 /nobreak >nul

echo Starting RoBERTa Backend Service...
start "RoBERTa-Backend" cmd /c "start-roberta-backend.bat"

timeout /t 3 /nobreak >nul

echo Starting Frontend...
start "Frontend" cmd /c "start-frontend.bat"

echo.
echo All services are starting...
echo.
echo Access the application at:
echo Frontend: Open frontend/index.html in your browser
echo AI-Backend API: http://localhost:8001
echo RoBERTa Backend API: http://localhost:8002
echo.
echo Press any key to exit...

pause
