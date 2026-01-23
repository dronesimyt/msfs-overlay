@echo off
setlocal EnableExtensions

REM === same paths as your PS1 ===
set "AppDir=%USERPROFILE%\source\repos\msfs-overlay"
set "PyExe=%LOCALAPPDATA%\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python3.13.exe"
set "CaddyExe=%LOCALAPPDATA%\Microsoft\WinGet\Packages\CaddyServer.Caddy_Microsoft.Winget.Source_8wekyb3d8bbwe\caddy.exe"
set "CaddyFile=%LOCALAPPDATA%\Caddy\Caddyfile"
REM ==============================

echo.
echo === DroneSim Overlay START ===
echo AppDir:   %AppDir%
echo PyExe:    %PyExe%
echo CaddyExe: %CaddyExe%
echo Caddyfile:%CaddyFile%
echo.

REM Start Flask hidden
echo Starting Flask...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
"Start-Process -WindowStyle Hidden -WorkingDirectory '%AppDir%' -FilePath '%PyExe%' -ArgumentList 'app.py'"

REM wait a bit
timeout /t 2 >nul

REM Start Caddy hidden
echo Starting Caddy...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
"Start-Process -WindowStyle Hidden -FilePath '%CaddyExe%' -ArgumentList 'run','--config','%CaddyFile%'"

echo.
echo Ready at https://overlay.dronesim.de
echo.

pause
