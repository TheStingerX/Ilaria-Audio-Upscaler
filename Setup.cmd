@echo off
setlocal
cd /d "%~dp0" || goto failed
set "app_exe=%~dp0Ilaria Audio Upcaler.exe"
set "python=App\.venv\Scripts\python.exe"
echo Installing Ilaria Audio Upscaler
echo Overall      [--------------------]   0%%  Preparing Python
if exist "%python%" goto runtime

echo [1/4] Creating the Python environment...
py -3.10-64 -m venv App\.venv
if not errorlevel 1 goto runtime
python -c "import struct,sys; sys.exit(sys.version_info[:2] != (3,10) or struct.calcsize('P') != 8)" >nul 2>&1
if errorlevel 1 goto python_missing
python -m venv App\.venv
if errorlevel 1 goto failed

:runtime
echo Overall      [#####---------------]  25%%  Python ready
nvidia-smi -L >nul 2>&1
if errorlevel 1 goto cpu
echo [2/4] NVIDIA GPU detected. Checking CUDA...
"%python%" -c "import torch, torchaudio; assert torch.cuda.is_available()" >nul 2>&1
if not errorlevel 1 goto dependencies
"%python%" "launcher\install_packages.py" cu128
if errorlevel 1 goto cuda_fallback
"%python%" -c "import torch, torchaudio; assert torch.cuda.is_available()" >nul 2>&1
if not errorlevel 1 goto dependencies

:cuda_fallback
echo CUDA could not be enabled. Continuing with CPU.
goto install_cpu

:cpu
echo [2/4] Checking the CPU runtime...
"%python%" -c "import torch, torchaudio" >nul 2>&1
if not errorlevel 1 goto dependencies

:install_cpu
"%python%" "launcher\install_packages.py" cpu
if errorlevel 1 goto failed

:dependencies
echo Overall      [##########----------]  50%%  PyTorch ready
echo [3/4] Installing app dependencies and FFmpeg...
"%python%" "launcher\install_packages.py" app
if errorlevel 1 goto failed

echo Overall      [###############-----]  75%%  Dependencies ready
if exist "%app_exe%" goto launch
echo [4/4] Building Ilaria Audio Upscaler.exe...
set "compiler=%WINDIR%\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if not exist "%compiler%" goto compiler_missing
"%compiler%" /nologo /target:winexe /platform:x64 /reference:System.Windows.Forms.dll /win32icon:"%~dp0launcher\App.ico" /out:"%app_exe%" "%~dp0launcher\Launcher.cs"
if errorlevel 1 goto failed
:launch
if not exist "%app_exe%" goto executable_missing
echo Overall      [####################] 100%%  Setup complete
"%python%" -c "import pathlib,subprocess,sys; exe=pathlib.Path(sys.argv[1]); subprocess.Popen([str(exe)], cwd=str(exe.parent))" "%app_exe%"
if errorlevel 1 goto launch_failed
exit /b 0

:executable_missing
echo The launcher was not created at "%app_exe%".
goto failed

:launch_failed
echo Setup finished. Open Ilaria Audio Upcaler.exe from this folder.
pause
exit /b 1

:python_missing
echo Install 64-bit Python 3.10 from https://www.python.org/downloads/ and run Setup.cmd again.
goto failed

:compiler_missing
echo The Windows .NET Framework C# compiler is unavailable.
goto failed

:failed
echo Setup could not finish.
pause
exit /b 1

