@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0EXTRACT-PACKAGE.ps1"
if errorlevel 1 (
  echo The full Strata package could not be extracted.
  pause
  exit /b 1
)
call "%~dp0.strata-full\Strata\INSTALL.bat" %*
exit /b %errorlevel%
