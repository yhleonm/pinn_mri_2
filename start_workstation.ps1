# Script de Arranque Dual para la Estacion PACS Electro-Termica 2.0 (FastAPI GPU + React Vite)
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  Iniciando Estacion PACS Electro-Termica 2.0 (ElectroThermalMRIPINN)          " -ForegroundColor Cyan
Write-Host "  Backend GPU (FastAPI + PyTorch CUDA): http://localhost:8002                   " -ForegroundColor Green
Write-Host "  Frontend PACS 2.0 (React + Vite):     http://localhost:5174                   " -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan

Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$PSScriptRoot'; python -m uvicorn backend.main:app --host 0.0.0.0 --port 8002 --reload"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$PSScriptRoot\frontend'; npm run dev"
