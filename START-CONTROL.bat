@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0EXTRACT-PACKAGE.ps1"
if errorlevel 1 exit /b 1
call "%~dp0.strata-full\START-CONTROL.bat" %*
