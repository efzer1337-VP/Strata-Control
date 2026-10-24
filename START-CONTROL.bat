@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0EXTRACT-PACKAGE.ps1"
if errorlevel 1 exit /b 1
if not exist "%~dp0.strata-full\Strata\.venv\Scripts\python.exe" (
  call "%~dp0INSTALL.bat" %*
  exit /b
)
call "%~dp0.strata-full\Strata\STRATA-CONTROL.bat" %*
