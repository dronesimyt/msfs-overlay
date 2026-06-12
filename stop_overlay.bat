@echo off
setlocal EnableExtensions

echo.
echo === DroneSim Overlay STOP ===
echo.

echo Stopping Flask...
powershell -Command "try { Invoke-WebRequest -Uri 'http://127.0.0.1:5000/shutdown' -Method POST -UseBasicParsing | Out-Null; Write-Host 'Flask stopped.' } catch { Write-Host 'Flask was not running.' }"
echo.


echo Done. Press any key to close.
pause >nul
