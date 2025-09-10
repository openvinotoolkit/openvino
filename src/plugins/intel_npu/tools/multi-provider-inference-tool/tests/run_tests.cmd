@echo off

:: Copyright (C) 2025 Intel Corporation
:: SPDX-License-Identifier: Apache-2.0

set SCRIPT_NAME=%~nx0
set SCRIPT_LOCATION=%~dp0
setlocal enabledelayedexpansion

set TEST_ENV_VENV_PATH=
set CLEAR_TEST_ENV=1
if not "%1"=="" (
    set TEST_ENV_VENV_PATH=%1
    set CLEAR_TEST_ENV=0
)


:: generate tmp venv to install requirements and execute tests
For /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c-%%a-%%b)
For /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a)
IF [%TEST_ENV_VENV_PATH%] == [] (
    set TEST_ENV_VENV_PATH=test_venv_%mydate%_%mytime%
    :input_arguments_loop
    IF exist !TEST_ENV_VENV_PATH! ( set TEST_ENV_VENV_PATH=!TEST_ENV_VENV_PATH!_new&& goto input_arguments_loop )
)

echo Use %TEST_ENV_VENV_PATH% venv path for testing...
set ret=0
goto setup_venv

:clear_test_env
if %CLEAR_TEST_ENV% EQU 1 (
    echo Clear test environment: %TEST_ENV_VENV_PATH%
    rmdir /s/q %TEST_ENV_VENV_PATH%
)
exit /B %ret%

:setup_venv
IF not exist %TEST_ENV_VENV_PATH% (
    python -m venv %TEST_ENV_VENV_PATH% --system-site-packages
    Call "%TEST_ENV_VENV_PATH%\Scripts\activate"
    echo Installing necessary requirements, please wait...
    pip install --ignore-installed -r tests\requirements.txt 2>&1 | findstr "ERROR" && set ret=-1
    if %ret% NEQ 0 (
        echo Couldn't configure test environment: %TEST_ENV_VENV_PATH%. Abort
        goto clear_test_env
    )
)
Call "%TEST_ENV_VENV_PATH%\Scripts\activate"
echo Execute tests ...
python -m unittest
set ret=%errorlevel%
goto clear_test_env
