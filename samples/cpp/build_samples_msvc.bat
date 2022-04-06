@echo off

:: Copyright (C) 2018-2022 Intel Corporation
:: SPDX-License-Identifier: Apache-2.0

@setlocal
SETLOCAL EnableDelayedExpansion
set "ROOT_DIR=%~dp0"
FOR /F "delims=\" %%i IN ("%ROOT_DIR%") DO set SAMPLES_TYPE=%%~nxi

set "SAMPLE_BUILD_DIR=%USERPROFILE%\Documents\Intel\OpenVINO\inference_engine_%SAMPLES_TYPE%_samples_build"
set SAMPLE_INSTALL_DIR=

:: command line arguments parsing
:input_arguments_loop
if not "%1"=="" (
    if "%1"=="-b" (
        set SAMPLE_BUILD_DIR=%2
        shift
    ) else if "%1"=="-i" (
        set SAMPLE_INSTALL_DIR=%2
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
    if exist "%ROOT_DIR%\..\..\setupvars.bat" (
        call "%ROOT_DIR%\..\..\setupvars.bat"
    ) else (
         echo Failed to set the environment variables automatically    
         echo To fix, run the following command: ^<INSTALL_DIR^>\setupvars.bat
         echo where INSTALL_DIR is the OpenVINO installation directory.
         GOTO errorHandling
    )
)

if "%PROCESSOR_ARCHITECTURE%" == "AMD64" (
   set "PLATFORM=x64"
) else (
   set "PLATFORM=Win32"
)

if exist "%SAMPLE_BUILD_DIR%\CMakeCache.txt" del "%SAMPLE_BUILD_DIR%\CMakeCache.txt"

cd /d "%ROOT_DIR%" && cmake -E make_directory "%SAMPLE_BUILD_DIR%" && cd /d "%SAMPLE_BUILD_DIR%" && cmake -G "Visual Studio 16 2019" -A %PLATFORM% "%ROOT_DIR%"

echo.
echo ###############^|^| Build OpenVINO Runtime samples using MS Visual Studio (MSBuild.exe) ^|^|###############
echo.

echo cmake --build . --config Release
cmake --build . --config Release
if ERRORLEVEL 1 GOTO errorHandling

if NOT "%SAMPLE_INSTALL_DIR%"=="" cmake -DCMAKE_INSTALL_PREFIX="%SAMPLE_INSTALL_DIR%" -DCOMPONENT=samples_bin -P cmake_install.cmake

echo Done.
exit /b

:usage
echo Build OpenVINO Runtime samples
echo.
echo Options:
echo   -h                       Print the help message
echo   -b SAMPLE_BUILD_DIR      Specify the sample build directory
echo   -i SAMPLE_INSTALL_DIR    Specify the sample install directory
exit /b

:errorHandling
echo Error
