@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" -B "%~dp0run.py" %*
