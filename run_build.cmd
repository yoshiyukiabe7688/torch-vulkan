@echo off
rem Load the MSVC environment, then run build.py (builds vkrt with cargo, then the C++ shim).
rem Set TORCH_VULKAN_PYTHON to the python.exe of an environment that has PyTorch installed
rem (its headers and libs are used for the build). Defaults to "python" on PATH.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul
set DISTUTILS_USE_SDK=1
set MAX_JOBS=8
set PATH=%PATH%;%LOCALAPPDATA%\Microsoft\WinGet\Links;%USERPROFILE%\.cargo\bin
if "%TORCH_VULKAN_PYTHON%"=="" (set "PY=python") else (set "PY=%TORCH_VULKAN_PYTHON%")
"%PY%" "%~dp0build.py"
