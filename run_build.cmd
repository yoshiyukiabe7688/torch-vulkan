@echo off
rem MSVC 環境を読み込んで build.py を実行する。
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul
set DISTUTILS_USE_SDK=1
set MAX_JOBS=8
set PATH=%PATH%;C:\Users\<user>\AppData\Local\Microsoft\WinGet\Links
"C:\Users\<user>\sd-scripts-no-cuda\venv\Scripts\python.exe" "%~dp0build.py"
