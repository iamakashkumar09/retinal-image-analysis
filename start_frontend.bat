@echo off
title Retinal DIP Frontend (Next.js)
echo Starting Next.js Frontend on http://localhost:3000 ...
cd /d "%~dp0retinal-dip\frontend"
"C:\Program Files\nodejs\npm.cmd" run dev
pause

