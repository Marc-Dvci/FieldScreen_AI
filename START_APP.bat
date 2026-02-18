@echo off
setlocal enabledelayedexpansion
echo ============================================
echo   FieldScreen AI — TB Screening Demo
echo   Starting Gradio app on localhost:7860
echo ============================================
echo.

REM ── Use the project's own virtual environment ──
set "VENV_PYTHON=%~dp0venv\Scripts\python.exe"
set "SCRIPT=%~dp0app.py"

if not exist "%VENV_PYTHON%" (
    echo ERROR: Virtual environment not found.
    echo        Run SETUP.bat first to create it.
    echo.
    pause
    exit /b 1
)

echo Using Python: %VENV_PYTHON%
echo.

REM ── Ensure CUDA DLLs are findable by llama-server ──
REM PyTorch bundles CUDA runtime DLLs in torch\lib
set "TORCH_LIB=%~dp0venv\Lib\site-packages\torch\lib"
if exist "%TORCH_LIB%" (
    set "PATH=%TORCH_LIB%;%PATH%"
    echo Added torch CUDA libs to PATH.
)
REM Also check nvidia packages (some PyTorch builds use these)
for /d %%d in ("%~dp0venv\Lib\site-packages\nvidia\*") do (
    if exist "%%d\bin" set "PATH=%%d\bin;!PATH!"
    if exist "%%d\lib" set "PATH=%%d\lib;!PATH!"
)
echo.

"%VENV_PYTHON%" "%SCRIPT%"

echo.
echo ============================================
echo   App stopped.
echo ============================================
pause
