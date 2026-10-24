@echo off
setlocal DisableDelayedExpansion
title Strata + Control - installation
cd /d "%~dp0"
if not exist "%~dp0control_install.py" goto incomplete
if exist "%~dp0.venv\Scripts\python.exe" goto install
call powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0INSTALL-PYTHON.ps1"
if errorlevel 1 exit /b 1
:install
set "PYTHONUTF8=1"
"%~dp0.venv\Scripts\python.exe" "%~dp0control_install.py" %*
if errorlevel 1 goto failed
exit /b 0
:incomplete
echo Extract the entire repository before running INSTALL.bat.
:failed
echo Installation failed. See the error above and run INSTALL.bat again.
pause
exit /b 1
