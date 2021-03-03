:: Copyright (C) 2018-2021 Intel Corporation
:: SPDX-License-Identifier: Apache-2.0

@echo off
setlocal enabledelayedexpansion

set TARGET=CPU
set BUILD_FOLDER=%USERPROFILE%\Documents\Intel\OpenVINO

:: command line arguments parsing
:input_arguments_loop
if not "%1"=="" (
    if "%1"=="-d" (
        set TARGET=%2
        echo target = !TARGET!
        shift
    )
    if "%1"=="-sample-options" (
        set SAMPLE_OPTIONS=%2 %3 %4 %5 %6
        echo sample_options = !SAMPLE_OPTIONS!
        shift
    )
    if "%1"=="-help" (
        echo %~n0%~x0 is classification demo using public SqueezeNet topology
        echo.
        echo Options:
        echo -d name     Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD are acceptable. Sample will look for a suitable plugin for device specified
        exit /b
    )
    shift
    goto :input_arguments_loop
)

set ROOT_DIR=%~dp0

set TARGET_PRECISION=FP16

echo target_precision = !TARGET_PRECISION!

set models_path=%BUILD_FOLDER%\openvino_models\models
set models_cache=%BUILD_FOLDER%\openvino_models\cache
set irs_path=%BUILD_FOLDER%\openvino_models\ir

set model_name=squeezenet1.1

set target_image_path=%ROOT_DIR%car.png

if exist "%ROOT_DIR%..\..\bin\setupvars.bat" (
    call "%ROOT_DIR%..\..\bin\setupvars.bat"
) else (
    echo setupvars.bat is not found, INTEL_OPENVINO_DIR can't be set
    goto error
)

echo INTEL_OPENVINO_DIR is set to %INTEL_OPENVINO_DIR%

:: Check if Python is installed
python --version 2>NUL
if errorlevel 1 (
    echo Error^: Python is not installed. Please install Python 3.5 ^(64-bit^) or higher from https://www.python.org/downloads/
    goto error
)

:: Check if Python version is equal or higher 3.4
for /F "tokens=* USEBACKQ" %%F IN (`python --version 2^>^&1`) DO (
    set version=%%F
)
echo %var%

for /F "tokens=1,2,3 delims=. " %%a in ("%version%") do (
    set Major=%%b
    set Minor=%%c
)

if "%Major%" geq "3" (
    if "%Minor%" geq "5" (
        set python_ver=okay
    )
)

if not "%python_ver%"=="okay" (
    echo Unsupported Python version. Please install Python 3.5 ^(64-bit^) or higher from https://www.python.org/downloads/
    goto error
)

:: install yaml python modules required for downloader.py
pip3 install --user -r "%ROOT_DIR%..\open_model_zoo\tools\downloader\requirements.in"
if ERRORLEVEL 1 GOTO errorHandling

set downloader_dir=%INTEL_OPENVINO_DIR%\deployment_tools\open_model_zoo\tools\downloader

for /F "tokens=* usebackq" %%d in (
    `python "%downloader_dir%\info_dumper.py" --name "%model_name%" ^|
        python -c "import sys, json; print(json.load(sys.stdin)[0]['subdirectory'])"`
) do (
    set model_dir=%%d
)

set ir_dir=%irs_path%\%model_dir%\%target_precision%

echo Download public %model_name% model
echo python "%downloader_dir%\downloader.py" --name "%model_name%" --output_dir "%models_path%" --cache_dir "%models_cache%"
python "%downloader_dir%\downloader.py" --name "%model_name%" --output_dir "%models_path%" --cache_dir "%models_cache%"
echo %model_name% model downloading completed

CALL :delay 7

if exist "%ir_dir%" (
    echo.
    echo Target folder %ir_dir% already exists. Skipping IR generation with Model Optimizer.
    echo If you want to convert a model again, remove the entire %ir_dir% folder.
    CALL :delay 7
    GOTO buildSample
)

echo.
echo ###############^|^| Install Model Optimizer prerequisites ^|^|###############
echo.
CALL :delay 3
cd /d "%INTEL_OPENVINO_DIR%\deployment_tools\model_optimizer\install_prerequisites"
call install_prerequisites_caffe.bat
if ERRORLEVEL 1 GOTO errorHandling

CALL :delay 7
echo.
echo ###############^|^| Run Model Optimizer ^|^|###############
echo.
CALL :delay 3

::set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
echo python "%downloader_dir%\converter.py" --mo "%INTEL_OPENVINO_DIR%\deployment_tools\model_optimizer\mo.py" --name "%model_name%" -d "%models_path%" -o "%irs_path%" --precisions "%TARGET_PRECISION%"
python "%downloader_dir%\converter.py" --mo "%INTEL_OPENVINO_DIR%\deployment_tools\model_optimizer\mo.py" --name "%model_name%" -d "%models_path%" -o "%irs_path%" --precisions "%TARGET_PRECISION%"
if ERRORLEVEL 1 GOTO errorHandling

CALL :delay 7

:buildSample
echo.
echo ###############^|^| Generate VS solution for Inference Engine samples using cmake ^|^|###############
echo.
CALL :delay 3

if "%PROCESSOR_ARCHITECTURE%" == "AMD64" (
    set "PLATFORM=x64"
) else (
    set "PLATFORM=Win32"
)

set VSWHERE="false"
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
    set VSWHERE="true"
    cd /d "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer"
) else if exist "%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe" (
    set VSWHERE="true"
    cd /d "%ProgramFiles%\Microsoft Visual Studio\Installer"
) else (
    echo "vswhere tool is not found"
)

if !VSWHERE! == "true" (
    for /f "usebackq tokens=*" %%i in (`vswhere -latest -products * -requires Microsoft.Component.MSBuild -property installationPath`) do (
        set VS_PATH=%%i
    )
    if exist "!VS_PATH!\MSBuild\14.0\Bin\MSBuild.exe" (
        set "MSBUILD_BIN=!VS_PATH!\MSBuild\14.0\Bin\MSBuild.exe"
    )
    if exist "!VS_PATH!\MSBuild\15.0\Bin\MSBuild.exe" (
        set "MSBUILD_BIN=!VS_PATH!\MSBuild\15.0\Bin\MSBuild.exe"
    )
    if exist "!VS_PATH!\MSBuild\Current\Bin\MSBuild.exe" (
        set "MSBUILD_BIN=!VS_PATH!\MSBuild\Current\Bin\MSBuild.exe"
    )
    for /f "usebackq tokens=1 delims=." %%i in (`vswhere -latest -products * -requires Microsoft.Component.MSBuild -property installationVersion`) do (
        set VS_MAJOR_VER=%%i
    )
    if "!VS_MAJOR_VER!"=="16" set "MSBUILD_VERSION=16 2019"
    if "!VS_MAJOR_VER!"=="15" set "MSBUILD_VERSION=15 2017"
)

if "!MSBUILD_BIN!" == "" (
    if exist "C:\Program Files (x86)\MSBuild\14.0\Bin\MSBuild.exe" (
        set "MSBUILD_BIN=C:\Program Files (x86)\MSBuild\14.0\Bin\MSBuild.exe"
        set "MSBUILD_VERSION=14 2015"
    )
    if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\MSBuild\15.0\Bin\MSBuild.exe" (
        set "MSBUILD_BIN=C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\MSBuild\15.0\Bin\MSBuild.exe"
        set "MSBUILD_VERSION=15 2017"
    )
    if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\MSBuild\15.0\Bin\MSBuild.exe" (
        set "MSBUILD_BIN=C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\MSBuild\15.0\Bin\MSBuild.exe"
        set "MSBUILD_VERSION=15 2017"
    )
    if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\MSBuild\15.0\Bin\MSBuild.exe" (
        set "MSBUILD_BIN=C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\MSBuild\15.0\Bin\MSBuild.exe"
        set "MSBUILD_VERSION=15 2017"
    )
) else (
    if not "!MSBUILD_BIN:2019=!"=="!MSBUILD_BIN!" set "MSBUILD_VERSION=16 2019"
    if not "!MSBUILD_BIN:2017=!"=="!MSBUILD_BIN!" set "MSBUILD_VERSION=15 2017"
    if not "!MSBUILD_BIN:2015=!"=="!MSBUILD_BIN!" set "MSBUILD_VERSION=14 2015"
)

if "!MSBUILD_BIN!" == "" (
    echo Build tools for Visual Studio 2015 / 2017 / 2019 cannot be found. If you use Visual Studio 2017, please download and install build tools from https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017
    GOTO errorHandling
)

set "SOLUTION_DIR64=%BUILD_FOLDER%\inference_engine_samples_build"

echo Creating Visual Studio !MSBUILD_VERSION! %PLATFORM% files in %SOLUTION_DIR64%... && ^
if exist "%SOLUTION_DIR64%\CMakeCache.txt" del "%SOLUTION_DIR64%\CMakeCache.txt"
cd /d "%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\samples\cpp" && cmake -E make_directory "%SOLUTION_DIR64%" && cd /d "%SOLUTION_DIR64%" && cmake -G "Visual Studio !MSBUILD_VERSION!" -A %PLATFORM% "%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\samples\cpp"
if ERRORLEVEL 1 GOTO errorHandling

CALL :delay 7

echo.
echo ###############^|^| Build Inference Engine samples using MS Visual Studio (MSBuild.exe) ^|^|###############
echo.
CALL :delay 3
echo "!MSBUILD_BIN!" Samples.sln /p:Configuration=Release /t:cpp_samples\classification_sample_async /clp:ErrorsOnly /m
"!MSBUILD_BIN!" Samples.sln /p:Configuration=Release /t:cpp_samples\classification_sample_async /clp:ErrorsOnly /m

if ERRORLEVEL 1 GOTO errorHandling

CALL :delay 7

:runSample
echo.
echo ###############^|^| Run Inference Engine classification sample ^|^|###############
echo.
CALL :delay 3
copy /Y "%ROOT_DIR%%model_name%.labels" "%ir_dir%"
cd /d "%SOLUTION_DIR64%\intel64\Release"
if not exist classification_sample_async.exe (
   cd /d "%INTEL_OPENVINO_DIR%\inference_engine\samples\cpp\intel64\Release"
)
echo classification_sample_async.exe -i "%target_image_path%" -m "%ir_dir%\%model_name%.xml" -d !TARGET! !SAMPLE_OPTIONS!
classification_sample_async.exe -i "%target_image_path%" -m "%ir_dir%\%model_name%.xml" -d !TARGET! !SAMPLE_OPTIONS!

if ERRORLEVEL 1 GOTO errorHandling

echo.
echo ###############^|^| Classification demo completed successfully ^|^|###############

CALL :delay 10
cd /d "%ROOT_DIR%"

goto :eof

:errorHandling
echo Error
cd /d "%ROOT_DIR%"

:delay
timeout %~1 2> nul
EXIT /B 0
