@echo off

:: Copyright (C) 2018-2026 Intel Corporation
:: SPDX-License-Identifier: Apache-2.0

set SCRIPT_NAME=%~nx0

set "INTEL_OPENVINO_DIR=%~dp0"

set "python_version="

:: command line arguments parsing
:input_arguments_loop
if not "%1"=="" (
    if "%1"=="-pyver" (
        set "python_version=%2"
        shift
    )
    shift
    goto :input_arguments_loop
)

:: OpenCV
if exist "%INTEL_OPENVINO_DIR%\opencv\setupvars.bat" (
   call "%INTEL_OPENVINO_DIR%\opencv\setupvars.bat"
   goto :opencv_done
)

if exist "%INTEL_OPENVINO_DIR%\extras\opencv\setupvars.bat" (
   call "%INTEL_OPENVINO_DIR%\extras\opencv\setupvars.bat"
   goto :opencv_done
)
:opencv_done

:: OpenVINO runtime
set "OpenVINO_DIR=%INTEL_OPENVINO_DIR%\runtime\cmake"
if exist "%OpenVINO_DIR%\OpenVINOGenAIConfig.cmake" (
   :: If GenAI is installed, export it as well.
   set "OpenVINOGenAI_DIR=%OpenVINO_DIR%"
)
set "OPENVINO_LIB_PATHS=%INTEL_OPENVINO_DIR%\runtime\bin\intel64\Release;%INTEL_OPENVINO_DIR%\runtime\bin\intel64\Debug;%OPENVINO_LIB_PATHS%"

:: TBB
if exist %INTEL_OPENVINO_DIR%\runtime\3rdparty\tbb (

   if exist %INTEL_OPENVINO_DIR%\runtime\3rdparty\tbb\redist (
      set "OPENVINO_LIB_PATHS=%INTEL_OPENVINO_DIR%\runtime\3rdparty\tbb\redist\intel64\vc14;%OPENVINO_LIB_PATHS%"
   ) else if exist %INTEL_OPENVINO_DIR%\runtime\3rdparty\tbb\bin\intel64\vc14 (
      set "OPENVINO_LIB_PATHS=%INTEL_OPENVINO_DIR%\runtime\3rdparty\tbb\bin\intel64\vc14;%OPENVINO_LIB_PATHS%"
   ) else if exist %INTEL_OPENVINO_DIR%\runtime\3rdparty\tbb\bin (
      set "OPENVINO_LIB_PATHS=%INTEL_OPENVINO_DIR%\runtime\3rdparty\tbb\bin;%OPENVINO_LIB_PATHS%"
   )

   if exist %INTEL_OPENVINO_DIR%\runtime\3rdparty\tbb\cmake (
      set "TBB_DIR=%INTEL_OPENVINO_DIR%\runtime\3rdparty\tbb\cmake"
   ) else if exist %INTEL_OPENVINO_DIR%\runtime\3rdparty\tbb\lib\cmake\TBB (
      set "TBB_DIR=%INTEL_OPENVINO_DIR%\runtime\3rdparty\tbb\lib\cmake\TBB"
   ) else if exist %INTEL_OPENVINO_DIR%\runtime\3rdparty\tbb\lib64\cmake\TBB (
      set "TBB_DIR=%INTEL_OPENVINO_DIR%\runtime\3rdparty\tbb\lib64\cmake\TBB"
   ) else if exist %INTEL_OPENVINO_DIR%\runtime\3rdparty\tbb\lib\cmake\tbb (
      set "TBB_DIR=%INTEL_OPENVINO_DIR%\runtime\3rdparty\tbb\lib\cmake\tbb"
   )
)

:: Add libs dirs to the PATH
set "PATH=%OPENVINO_LIB_PATHS%;%PATH%"

:: Check if Python is installed
python --version 2>NUL
if errorlevel 1 (call :python_not_installed) else (call :check_python_version)

echo [setupvars.bat] OpenVINO environment initialized

exit /B 0

:python_not_installed
call :get_available_python_versions
if "%available_python_versions%"=="" (
   echo Warning^: Python is not installed. Please install Python ^(64-bit^) from https://www.python.org/downloads/
) else (
   echo Warning^: Python is not installed. Please install one of Python%available_python_versions% ^(64-bit^) from https://www.python.org/downloads/
)
exit /B 0

:check_python_version
:: Check Python version if user did not pass -pyver
if "%python_version%" == "" (
    for /F "tokens=* USEBACKQ" %%F IN (`python -c "import sys; print(str(sys.version_info[0])+'.'+str(sys.version_info[1]))" 2^>^&1`) DO (
       set python_version=%%F
    )
)

for /F "tokens=1,2 delims=. " %%a in ("%python_version%") do (
   set pyversion_major=%%a
   set pyversion_minor=%%b
)

:: Strip non-numeric suffix from minor version (e.g., 14t -> 14)
for /f "delims=t" %%i in ("%pyversion_minor%") do set "pyversion_minor=%%i"

set "current_python_version=%pyversion_major%.%pyversion_minor%"

call :get_available_python_versions
if "%available_python_versions%"=="" (
   echo Warning^: Could not detect which Python versions the OpenVINO Python API in this package was built for. The installed Python version will not be verified.
   goto :python_version_ok
)

set "check_pyversion="
for %%v in (%available_python_versions%) do call :match_python_version "%%v"
if "%check_pyversion%"=="true" goto :python_version_ok

echo Unsupported Python version %current_python_version%. The OpenVINO Python API in this package is built for Python:%available_python_versions%. Please activate a Python environment with a matching version.
exit /B 0

:python_version_ok
:: Check Python bitness
python -c "import sys; print(64 if sys.maxsize > 2**32 else 32)" 2 > NUL
if errorlevel 1 (
   echo Warning^: Cannot determine installed Python bitness
   exit /B 0
)

for /F "tokens=* USEBACKQ" %%F IN (`python -c "import sys; print(64 if sys.maxsize > 2**32 else 32)" 2^>^&1`) DO (
   set bitness=%%F
)

if not "%bitness%"=="64" (
   echo Unsupported Python bitness. Please install a 64-bit Python interpreter from https://www.python.org/downloads/
   exit /B 0
)

set PYTHONPATH=%INTEL_OPENVINO_DIR%\python;%INTEL_OPENVINO_DIR%\python\python3;%PYTHONPATH%
exit /B 0

:get_available_python_versions
set "available_python_versions="
set "ov_python_dir=%INTEL_OPENVINO_DIR%\python\openvino"
if not exist "%ov_python_dir%" exit /B 0
if not exist "%ov_python_dir%\_pyopenvino.cp*.pyd" exit /B 0
for %%F in ("%ov_python_dir%\_pyopenvino.cp*.pyd") do call :parse_pyd_version "%%~nxF"
exit /B 0

:parse_pyd_version
:: %1 is a file name like _pyopenvino.cp312-win_amd64.pyd
set "fname=%~1"
set "tag=%fname:*_pyopenvino.cp=%"
for /f "delims=-" %%A in ("%tag%") do set "vernum=%%A"
call :strip_suffix vernum
set "ver=%vernum:~0,1%.%vernum:~1%"
echo %available_python_versions% | findstr /C:" %ver%" >NUL 2>&1 || set "available_python_versions=%available_python_versions% %ver%"
set "ver="
exit /B 0

:match_python_version
if "%~1"=="%current_python_version%" set "check_pyversion=true"
exit /B 0

:strip_suffix
:: Remove non-numeric suffix from a version number variable
:: Usage: call :strip_suffix variable_name
:: Strip 't' suffix (e.g., 14t -> 14)
setlocal enabledelayedexpansion
set "var_value=!%~1!"
for /f "delims=t" %%i in ("!var_value!") do set "var_value=%%i"
endlocal & set "%~1=%var_value%"
exit /B 0

:GetFullPath
SET %2=%~f1

GOTO :EOF
