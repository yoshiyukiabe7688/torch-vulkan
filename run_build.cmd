@echo off
rem Load the MSVC environment, then run build.py (builds vkrt with cargo, then the C++ shim).
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul
set DISTUTILS_USE_SDK=1
set MAX_JOBS=8
set PATH=%PATH%;C:\Users\<user>\AppData\Local\Microsoft\WinGet\Links;C:\Users\<user>\.cargo\bin
"C:\Users\<user>\sd-scripts-no-cuda\venv\Scripts\python.exe" "%~dp0build.py"
