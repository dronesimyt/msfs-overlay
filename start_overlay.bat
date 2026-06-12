@echo off
setlocal EnableExtensions

net session >nul 2>&1
if %errorLevel% NEQ 0 (
  echo Requesting admin rights...
  powershell -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)

set "AppDir=%USERPROFILE%\source\repos\msfs-overlay"
set "PyExe=%LOCALAPPDATA%\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python3.13.exe"

echo.
echo === DroneSim Overlay START ===
echo.

REM Kill any previous Flask instance
taskkill /IM python3.13.exe /F >nul 2>&1

REM Start Flask hidden
echo Starting Flask...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "Start-Process -WindowStyle Hidden -WorkingDirectory '%AppDir%' -FilePath '%PyExe%' -ArgumentList 'app.py'"

timeout /t 2 >nul

echo.
echo Overlay: https://overlay.dronesim.de/overlay
echo.

pause
