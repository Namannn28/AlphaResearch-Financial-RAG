@echo off
echo ============================================================
echo   CandidateFit-RAG: Launching Web Dashboard
echo ============================================================
echo Activating Python virtual environment...
call venv\Scripts\activate.bat

echo Starting Streamlit app on http://localhost:8501 ...
streamlit run app.py
pause
