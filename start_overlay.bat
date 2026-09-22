@echo off
setlocal EnableExtensions

REM Project folder = folder of this script
set "AppDir=%~dp0"
if "%AppDir:~-1%"=="\" set "AppDir=%AppDir:~0,-1%"

REM Python: scoop install if present, otherwise whatever is on PATH
set "PyExe=%USERPROFILE%\scoop\apps\python\current\python.exe"
if not exist "%PyExe%" (
  for /f "delims=" %%P in ('where python 2^>nul') do if not defined PyFound set "PyFound=%%P"
)
if defined PyFound set "PyExe=%PyFound%"
if not exist "%PyExe%" (
  echo Python not found. Install Python or add it to PATH.
  pause
  exit /b 1
)

echo.
echo === DroneSim Overlay START ===
echo.

REM Stop a previous overlay instance only (no other Python processes)
powershell.exe -NoProfile -Command ^
  "try { Invoke-WebRequest -Uri 'http://127.0.0.1:5000/shutdown' -Method POST -UseBasicParsing -TimeoutSec 3 | Out-Null; Write-Host 'Previous instance stopped.'; Start-Sleep -Milliseconds 500 } catch {}"

REM Start Flask hidden
echo Starting Flask...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "Start-Process -WindowStyle Hidden -WorkingDirectory '%AppDir%' -FilePath '%PyExe%' -ArgumentList '-B app.py'"

timeout /t 2 >nul

echo.
echo Overlay: https://overlay.dronesim.de/
echo.
