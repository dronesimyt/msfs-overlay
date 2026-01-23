@echo off
setlocal EnableExtensions

echo.
echo === DroneSim Overlay STOP ===
echo.

echo Killing Python (python3.13.exe)...
taskkill /IM python3.13.exe /F
echo ExitCode (python): %ERRORLEVEL%
echo.

echo Killing Caddy (caddy.exe)...
taskkill /IM caddy.exe /F
echo ExitCode (caddy): %ERRORLEVEL%
echo.

echo Done. Press any key to close.
pause >nul
