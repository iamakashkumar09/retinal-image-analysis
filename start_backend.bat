@echo off
title Retinal DIP Backend (FastAPI)
echo Starting Retinal Image Analysis Python API on http://localhost:8000 ...
cd /d "%~dp0retinal-dip"
"C:\Users\Akash\AppData\Local\Programs\Python\Python311\Scripts\uvicorn.exe" api:app --reload --port 8000
pause

