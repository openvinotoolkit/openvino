:: Copyright (C) 2018-2022 Intel Corporation
:: SPDX-License-Identifier: Apache-2.0

@echo off
setlocal enabledelayedexpansion

set ROOT_DIR=%~dp0

set TARGET=CPU
set BUILD_FOLDER=%USERPROFILE%\Documents\Intel\OpenVINO

:: command line arguments parsing
:input_arguments_loop
if not "%1"=="" (
    if "%1"=="-b" (
        set BUILD_FOLDER=%2
        shift
    )
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
        echo Benchmark sample using public SqueezeNet topology
        echo.
        echo Options:
        echo    -help                      Print help message
        echo    -b BUILD_FOLDER            Specify the sample build directory
        echo    -d DEVICE                  Specify the target device to infer on; CPU, GPU, HDDL or MYRIAD are acceptable. Sample will look for a suitable plugin for device specified
        echo    -sample-options OPTIONS    Specify command line arguments for the sample
        exit /b
    )
    shift
    goto :input_arguments_loop
)

set "SOLUTION_DIR64=%BUILD_FOLDER%\inference_engine_cpp_samples_build"

IF "%SAMPLE_OPTIONS%"=="" (
    set SAMPLE_OPTIONS=-niter 1000 
)

set TARGET_PRECISION=FP16

echo target_precision = !TARGET_PRECISION!

set models_path=%BUILD_FOLDER%\openvino_models\models
set models_cache=%BUILD_FOLDER%\openvino_models\cache
set irs_path=%BUILD_FOLDER%\openvino_models\ir

set model_name=squeezenet1.1

set target_image_path=%ROOT_DIR%car.png

set omz_tool_error_message=It is required to download and convert a model. Check https://pypi.org/project/openvino-dev/ to install it. Then run the script again.

if exist "%ROOT_DIR%..\..\setupvars.bat" (
    call "%ROOT_DIR%..\..\setupvars.bat"
) else (
    echo setupvars.bat is not found, INTEL_OPENVINO_DIR can't be set
    goto error
)

echo INTEL_OPENVINO_DIR is set to %INTEL_OPENVINO_DIR%

:: Check if Python is installed
python --version 2>NUL
if errorlevel 1 (
    echo Error^: Python is not installed. Please install Python 3.6 ^(64-bit^) or higher from https://www.python.org/downloads/
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
    if "%Minor%" geq "6" (
        set python_ver=okay
    )
)

if not "%python_ver%"=="okay" (
    echo Unsupported Python version. Please install Python 3.6 ^(64-bit^) or higher from https://www.python.org/downloads/
    goto error
)

omz_info_dumper --print_all >NUL
if errorlevel 1 (
    echo Error: omz_info_dumper was not found. %omz_tool_error_message%
    goto error
)

omz_downloader --print_all >NUL
if errorlevel 1 (
    echo Error: omz_downloader was not found. %omz_tool_error_message%
    goto error
)

omz_converter --print_all >NUL
if errorlevel 1 (
    echo Error: omz_converter was not found. %omz_tool_error_message%
    goto error
)

for /F "tokens=* usebackq" %%d in (
    `omz_info_dumper --name "%model_name%" ^|
        python -c "import sys, json; print(json.load(sys.stdin)[0]['subdirectory'])"`
) do (
    set model_dir=%%d
)

set ir_dir=%irs_path%\%model_dir%\%target_precision%

echo.
echo Download public %model_name% model
echo omz_downloader --name "%model_name%" --output_dir "%models_path%" --cache_dir "%models_cache%"
omz_downloader --name "%model_name%" --output_dir "%models_path%" --cache_dir "%models_cache%"
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
echo ###############^|^| Run Model Optimizer ^|^|###############
echo.
CALL :delay 3

::set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
echo omz_converter --mo "%INTEL_OPENVINO_DIR%\tools\model_optimizer\mo.py" --name "%model_name%" -d "%models_path%" -o "%irs_path%" --precisions "%TARGET_PRECISION%"
omz_converter --mo "%INTEL_OPENVINO_DIR%\tools\model_optimizer\mo.py" --name "%model_name%" -d "%models_path%" -o "%irs_path%" --precisions "%TARGET_PRECISION%"
if ERRORLEVEL 1 GOTO errorHandling

CALL :delay 7

:buildSample
echo.
echo ###############^|^| Generate VS solution for OpenVINO Runtime samples using cmake ^|^|###############
echo.
CALL :delay 3

if "%PROCESSOR_ARCHITECTURE%" == "AMD64" (
    set "PLATFORM=x64"
) else (
    set "PLATFORM=Win32"
)

if exist "%SOLUTION_DIR64%\CMakeCache.txt" del "%SOLUTION_DIR64%\CMakeCache.txt"

cd /d "%INTEL_OPENVINO_DIR%\samples\cpp" && cmake -E make_directory "%SOLUTION_DIR64%" && cd /d "%SOLUTION_DIR64%" && cmake -G "Visual Studio 16 2019" -A %PLATFORM% "%INTEL_OPENVINO_DIR%\samples\cpp"
if ERRORLEVEL 1 GOTO errorHandling

CALL :delay 7

echo.
echo ###############^|^| Build OpenVINO Runtime samples using cmake ^|^|###############
echo.

CALL :delay 3

echo cmake --build . --config Release --target benchmark_app
cmake --build . --config Release --target benchmark_app
if ERRORLEVEL 1 GOTO errorHandling

CALL :delay 7

:runSample
echo.
echo ###############^|^| Run OpenVINO Runtime benchmark app ^|^|###############
echo.
CALL :delay 3
copy /Y "%ROOT_DIR%%model_name%.labels" "%ir_dir%"
cd /d "%SOLUTION_DIR64%\intel64\Release"

echo benchmark_app.exe -i "%target_image_path%" -m "%ir_dir%\%model_name%.xml" -pc  -d  !TARGET! !SAMPLE_OPTIONS!
benchmark_app.exe -i "%target_image_path%" -m "%ir_dir%\%model_name%.xml" -pc  -d  !TARGET! !SAMPLE_OPTIONS!

if ERRORLEVEL 1 GOTO errorHandling

echo.
echo ###############^|^| OpenVINO Runtime benchmark app completed successfully ^|^|###############

CALL :delay 10
cd /d "%ROOT_DIR%"

goto :eof

:errorHandling
echo Error
cd /d "%ROOT_DIR%"

:delay
timeout %~1 2> nul
EXIT /B 0
