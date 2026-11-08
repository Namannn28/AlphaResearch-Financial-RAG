# CandidateFit-RAG PowerShell Startup Script
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  CandidateFit-RAG: Launching Interactive Web Dashboard" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$VenvPython = Join-Path $PSScriptRoot "venv\Scripts\python.exe"

if (-Not (Test-Path $VenvPython)) {
    Write-Host "Virtual environment not detected. Creating one now..." -ForegroundColor Yellow
    python -m venv venv
    & .\venv\Scripts\pip.exe install -r requirements.txt
}

Write-Host "Activating virtual environment and launching Streamlit on http://localhost:8501..." -ForegroundColor Green
& .\venv\Scripts\streamlit.exe run app.py
