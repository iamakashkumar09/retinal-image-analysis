@echo off
title Retinal Image Analysis (Streamlit App)
echo =========================================================
echo   Starting Retinal Image Analysis Web UI (Streamlit)
echo =========================================================
cd /d "%~dp0retinal-dip"
"C:\Users\Akash\AppData\Local\Programs\Python\Python311\Scripts\streamlit.exe" run app.py
pause

