@echo off
setlocal
cd /d "%~dp0"
title Match Trainer

if not exist ".venv\Scripts\python.exe" (
  echo Setting up Match Trainer for the first time...
  py -3.13 -m venv .venv || goto :setup_failed
)

".venv\Scripts\python.exe" -c "import numpy" 2>nul
if errorlevel 1 (
  echo Installing what Match Trainer needs. This uses the internet once.
  ".venv\Scripts\python.exe" -m pip install --quiet --disable-pip-version-check -r requirements.txt || goto :setup_failed
)

".venv\Scripts\python.exe" -m trainer
if errorlevel 1 (
  echo.
  echo Match Trainer stopped with an error. The lines above say what went wrong.
  pause
)
goto :eof

:setup_failed
echo.
echo Setup did not finish. Check that Python 3.13 is installed ^(python.org^) and that
echo this computer is online, then double-click this file again.
pause
