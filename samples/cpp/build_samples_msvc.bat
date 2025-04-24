@echo off

:: Copyright (C) 2018-2025 Intel Corporation
:: SPDX-License-Identifier: Apache-2.0

@setlocal
SETLOCAL EnableDelayedExpansion
set "SAMPLES_SOURCE_DIR=%~dp0"
FOR %%i IN ("%SAMPLES_SOURCE_DIR%\.") DO set SAMPLES_TYPE=%%~nxi

set "SAMPLES_BUILD_DIR=%USERPROFILE%\Documents\Intel\OpenVINO\openvino_%SAMPLES_TYPE%_samples_build"
set SAMPLES_INSTALL_DIR=

:: command line arguments parsing
:input_arguments_loop
if not "%1"=="" (
    if "%1"=="-b" (
        set SAMPLES_BUILD_DIR=%2
        shift
    ) else if "%1"=="-i" (
        set SAMPLES_INSTALL_DIR=%2
        shift
    ) else if "%1"=="-h" (
        goto usage
    ) else (
        echo Unrecognized option specified "%1"
        goto usage
    )
    shift
    goto :input_arguments_loop
)

if "%INTEL_OPENVINO_DIR%"=="" (
    if exist "%SAMPLES_SOURCE_DIR%\..\..\setupvars.bat" (
        call "%SAMPLES_SOURCE_DIR%\..\..\setupvars.bat"
    ) else (
        echo Failed to set the environment variables automatically. To fix, run the following command:
        echo ^<INTEL_OPENVINO_DIR^>\setupvars.bat
        echo where INTEL_OPENVINO_DIR is the OpenVINO installation directory
        exit /b 1
    )
)

if exist "%SAMPLES_BUILD_DIR%\CMakeCache.txt" del "%SAMPLES_BUILD_DIR%\CMakeCache.txt"

cd /d "%SAMPLES_SOURCE_DIR%" && cmake -E make_directory "%SAMPLES_BUILD_DIR%" && cd /d "%SAMPLES_BUILD_DIR%" && cmake -DCMAKE_DISABLE_FIND_PACKAGE_PkgConfig=ON "%SAMPLES_SOURCE_DIR%"
if ERRORLEVEL 1 GOTO errorHandling

echo.
echo ###############^|^| Build OpenVINO Runtime samples using MS Visual Studio (MSBuild.exe) ^|^|###############
echo.

echo cmake --build "%SAMPLES_BUILD_DIR%" --config Release --parallel
cmake --build "%SAMPLES_BUILD_DIR%" --config Release --parallel
if ERRORLEVEL 1 GOTO errorHandling

if NOT "%SAMPLES_INSTALL_DIR%"=="" (
    cmake -DCMAKE_INSTALL_PREFIX="%SAMPLES_INSTALL_DIR%" -DCOMPONENT=samples_bin -P "%SAMPLES_BUILD_DIR%\cmake_install.cmake"
)

echo Done.
exit /b

:usage
echo Build OpenVINO Runtime samples
echo.
echo Options:
echo   -h                        Print the help message
echo   -b SAMPLES_BUILD_DIR      Specify the samples build directory
echo   -i SAMPLES_INSTALL_DIR    Specify the samples install directory
exit /b

:errorHandling
echo Error
exit /b %ERRORLEVEL%
