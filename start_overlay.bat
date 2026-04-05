@echo off
setlocal EnableExtensions

set "AppDir=%USERPROFILE%\source\repos\msfs-overlay"
set "PyExe=%LOCALAPPDATA%\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python3.13.exe"
set "CaddyExe=%LOCALAPPDATA%\Microsoft\WinGet\Packages\CaddyServer.Caddy_Microsoft.Winget.Source_8wekyb3d8bbwe\caddy.exe"

REM Use the Caddyfile inside the repo (deterministic)
set "CaddyFile=%AppDir%\Caddyfile"
set "CaddyOut=%AppDir%\caddy.out.log"
set "CaddyErr=%AppDir%\caddy.err.log"

echo.
echo === DroneSim Overlay START ===
echo AppDir:    %AppDir%
echo PyExe:     %PyExe%
echo CaddyExe:  %CaddyExe%
echo CaddyFile: %CaddyFile%
echo.

REM Stop old Caddy so ports aren't blocked
taskkill /f /im caddy.exe >nul 2>&1

REM Start Flask hidden
echo Starting Flask...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "Start-Process -WindowStyle Hidden -WorkingDirectory '%AppDir%' -FilePath '%PyExe%' -ArgumentList 'app.py'"

REM give Flask a moment
timeout /t 2 >nul

REM Start Caddy hidden + logs (stdout/stderr must be different files)
echo Starting Caddy...
REM Start Caddy visible (for debugging)
start "Caddy" "%CaddyExe%" run --config "%CaddyFile%" --adapter caddyfile

echo.
echo Test local Flask:
echo   http://127.0.0.1:5000/
echo.
echo Test via Caddy:
echo   https://overlay.dronesim.de/
echo.
echo If Caddy still shows 404, open:
echo   %CaddyOut%
echo   %CaddyErr%
echo.

pause