@echo off

:: Copyright (C) 2025 Intel Corporation
:: SPDX-License-Identifier: Apache-2.0

set SCRIPT_NAME=%~nx0
set SCRIPT_LOCATION=%~dp0

set "python_version="
set OV_PACKAGE_DIR=
set PACKAGE_MODE_VENV_PATH=

setlocal enabledelayedexpansion
:: command line arguments parsing
:input_arguments_loop
if not "%1"=="" (
    if "%1"=="-pyver" (
        set "python_version=%2"
        shift
    ) else (
        if [!OV_PACKAGE_DIR!] == [] (
            set OV_PACKAGE_DIR=%1
        ) else (
            set PACKAGE_MODE_VENV_PATH=%1
        )
    )
    shift
    goto :input_arguments_loop
)

iF [%OV_PACKAGE_DIR%] == [] (
    echo OV_PACKAGE_DIR is not set: please specify it as a first commandline arguments of the script !SCRIPT_NAME! invocation
    exit /B -1
)
if not exist "%OV_PACKAGE_DIR%/setupvars.bat" (
    echo %OV_PACKAGE_DIR% must contain setupvars.bat
    exit /B -1
)
 
IF [%PACKAGE_MODE_VENV_PATH%] == [] (
    set PACKAGE_MODE_VENV_PATH=.venv_package_mode
    echo Use the default venv name for PACKAGE mode: !PACKAGE_MODE_VENV_PATH!
)

:: Check if Python is installed
set PYTHON_VERSION_MAJOR=3
set MIN_REQUIRED_PYTHON_VERSION_MINOR=9
set MAX_SUPPORTED_PYTHON_VERSION_MINOR=13

python --version 2>NUL
if errorlevel 1 (call :python_not_installed) else (call :check_python_version)
set rc=%ERRORLEVEL%
if %rc% NEQ 0 echo Could not initialize venv %PACKAGE_MODE_VENV_PATH%. Abort && exit /B %rc%
echo To enter the venv, please use "%PACKAGE_MODE_VENV_PATH%\Scripts\activate"
exit /B %rc%

:python_not_installed
echo Warning^: Python is not installed. Please install one of Python %PYTHON_VERSION_MAJOR%.%MIN_REQUIRED_PYTHON_VERSION_MINOR% - %PYTHON_VERSION_MAJOR%.%MAX_SUPPORTED_PYTHON_VERSION_MINOR% ^(64-bit^) from https://www.python.org/downloads/
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

if %pyversion_major% equ %PYTHON_VERSION_MAJOR% (
   if %pyversion_minor% geq %MIN_REQUIRED_PYTHON_VERSION_MINOR% (
      if %pyversion_minor% leq %MAX_SUPPORTED_PYTHON_VERSION_MINOR% (
         set "check_pyversion=true"
      )
   )   
)

if not "%check_pyversion%"=="true" (
   echo Unsupported Python version %pyversion_major%.%pyversion_minor%. Please install one of Python %PYTHON_VERSION_MAJOR%.%MIN_REQUIRED_PYTHON_VERSION_MINOR% - %PYTHON_VERSION_MAJOR%.%MAX_SUPPORTED_PYTHON_VERSION_MINOR% ^(64-bit^) from https://www.python.org/downloads/
   exit /B 0
)

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
   echo Unsupported Python bitness. Please install one of Python %PYTHON_VERSION_MAJOR%.%MIN_REQUIRED_PYTHON_VERSION_MINOR% - %PYTHON_VERSION_MAJOR%.%MAX_SUPPORTED_PYTHON_VERSION_MINOR%^(64-bit^) from https://www.python.org/downloads/
   exit /B 0
)

IF exist %PACKAGE_MODE_VENV_PATH% ( 
    echo Python venv for PACKAGE mode exists: %PACKAGE_MODE_VENV_PATH%. Skip it
) ELSE (
    copy /b/v/y requirements\generated_gathered_requirements_from_ov_package.txt requirements\generated_gathered_requirements_from_ov_package.txt_bk >NUL
    python requirements/find_inner_ov_package_requirements.py !OV_PACKAGE_DIR!
    python -m venv %PACKAGE_MODE_VENV_PATH%
    echo !OV_PACKAGE_DIR!/setupvars.bat >> %PACKAGE_MODE_VENV_PATH%\Scripts\activate.bat
    Call "%PACKAGE_MODE_VENV_PATH%\Scripts\activate"

    set ret=0
    echo Installing necessary requirements, please wait...
    pip install -r requirements\ov_package_mode_requirements.txt 2>&1 | findstr "ERROR" && set ret=-1
    copy /b/v/y requirements\generated_gathered_requirements_from_ov_package.txt_bk requirements\generated_gathered_requirements_from_ov_package.txt >NUL
    del requirements\generated_gathered_requirements_from_ov_package.txt_bk
    
    if !ret! NEQ 0 (
        rmdir /s/q %PACKAGE_MODE_VENV_PATH%
        exit /B !ret!
    )
    ::Setting up test environment as a custom post activation script
    echo %SCRIPT_LOCATION%\tests\run_tests.cmd .venv_tests >> "%PACKAGE_MODE_VENV_PATH%\Scripts\activate.bat"
    echo Python venv for PACKAGE mode initialized: %PACKAGE_MODE_VENV_PATH%
    exit /B !ret!
)

exit /B 0
