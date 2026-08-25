@echo off
rem Configure, build and run the standalone core tests (pure CMake ??? no manual
rem cl/link commands). Toolchain bootstrap (once):
rem   powershell -ExecutionPolicy Bypass -File tools\setup_vktools.ps1
setlocal
set SCRIPT_DIR=%~dp0

if not exist "%LOCALAPPDATA%\vktools\glslang\bin\glslang.exe" (
    if not defined VULKAN_SDK (
        echo No Vulkan SDK and no vktools - run:
        echo   powershell -ExecutionPolicy Bypass -File tools\setup_vktools.ps1
        exit /b 1
    )
)

set GLSLANG_HINT=
if exist "%LOCALAPPDATA%\vktools\glslang\bin\glslang.exe" set GLSLANG_HINT=-DGLSLANG="%LOCALAPPDATA%\vktools\glslang\bin\glslang.exe"

cmake -S "%SCRIPT_DIR%." -B "%SCRIPT_DIR%build" -G "Visual Studio 17 2022" -A x64 %GLSLANG_HINT% || exit /b 1
cmake --build "%SCRIPT_DIR%build" --config Release || exit /b 1

echo === running tests ===
setlocal enabledelayedexpansion
set FAILED=0
for %%f in ("%SCRIPT_DIR%build\Release\test_*.exe") do (
    "%%~ff"
    "%%~ff"
    if !errorlevel! neq 0 set FAILED=1
)
if %FAILED% neq 0 exit /b 1
echo ALL TEST EXECUTABLES PASS


