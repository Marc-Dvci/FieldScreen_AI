@echo off
setlocal enabledelayedexpansion

echo.
echo ============================================
echo   FieldScreen AI — Environment Setup
echo ============================================
echo.
echo   This creates a self-contained Python
echo   environment with all dependencies.
echo   Run this ONCE, then use START_APP.bat.
echo.
echo ============================================
echo.

REM ── Check Python ──
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH.
    echo.
    echo Install Python 3.10 or later from https://python.org
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version 2^>^&1') do echo Found: %%i
echo.

REM ── Create virtual environment ──
if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Done.
) else (
    echo Virtual environment already exists.
)
echo.

call venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet

REM ── 1. PyTorch with CUDA ──
echo [1/4] Installing PyTorch with CUDA 12.8...
echo       (this downloads ~2.5 GB on first run)
echo.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo.
    echo WARNING: PyTorch CUDA install failed.
    echo   If you have a different CUDA version, edit this script.
    echo   Available indexes: cu118, cu121, cu124, cu128
    echo   Example: --index-url https://download.pytorch.org/whl/cu121
    echo.
)

REM ── 2. Transformers from source (MedASR needs >= 5.0) ──
echo.
echo [2/4] Installing transformers from source (required for MedASR)...
echo.
pip install "git+https://github.com/huggingface/transformers.git"
if errorlevel 1 (
    echo.
    echo WARNING: transformers source install failed.
    echo   Trying release version as fallback...
    pip install "transformers>=4.45"
)

REM ── 3. Remaining Python dependencies ──
echo.
echo [3/4] Installing Python dependencies...
echo.
pip install -r requirements.txt

REM ── 4. llama-server binary ──
echo.
echo [4/4] Setting up llama-server...
if not exist "bin" mkdir bin

if exist "bin\llama-server.exe" (
    echo llama-server.exe already present in bin\
    goto :server_done
)

REM Try venv site-packages
for /r "venv\Lib\site-packages" %%f in (llama-server.exe) do (
    if exist "%%f" (
        echo Copying llama-server.exe from venv...
        copy "%%f" "bin\llama-server.exe" >nul
        echo Done.
        goto :server_done
    )
)

echo.
echo WARNING: llama-server.exe not found automatically.
echo.
echo   Please copy llama-server.exe into the bin\ directory.
echo   Download CUDA builds from:
echo     https://github.com/ggerganov/llama.cpp/releases
echo   (look for llama-...-bin-win-cuda-cu12...-x64.zip)
echo.

:server_done

REM ── Check for GGUF models ──
echo.
echo ── Model file check ──
if not exist "Models\MedGemma" mkdir "Models\MedGemma"

set "FOUND_GGUF=0"
for %%f in (Models\MedGemma\*.gguf) do set "FOUND_GGUF=1"

if "!FOUND_GGUF!"=="0" (
    echo.
    echo   Models\MedGemma\ is empty. For a portable setup, copy:
    echo     - medgemma-1.5-4b-it-Q4_K_M.gguf  (main model)
    echo     - mmproj-BF16.gguf                  (vision projector)
    echo   into Models\MedGemma\
    echo.
    echo   Or leave them where they are — app.py will check
    echo   the text-generation-webui paths as fallback.
) else (
    echo   Found GGUF models in Models\MedGemma\
)

echo.
echo ============================================
echo   Setup complete!
echo.
echo   To launch:  START_APP.bat
echo   To move to another machine:
echo     1. Copy this entire folder
echo     2. Run SETUP.bat on the new machine
echo     3. Copy GGUF models into Models\MedGemma\
echo ============================================
echo.
pause
