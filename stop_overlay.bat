@echo off
setlocal EnableExtensions

net session >nul 2>&1
if %errorlevel% neq 0 (
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

echo.
echo === DroneSim Overlay STOP ===
echo.

echo Killing Python (python3.13.exe)...
taskkill /IM python.exe /F
echo ExitCode (python): %ERRORLEVEL%
echo.


echo Done. Press any key to close.
pause >nul
