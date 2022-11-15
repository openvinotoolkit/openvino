@echo off

:: Copyright (C) 2018-2022 Intel Corporation
:: SPDX-License-Identifier: Apache-2.0

:: Check if Python is installed
setlocal

set ROOT_DIR=%~dp0

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

:: install Python modules

set USE_VENV="false"
set VENV_DIR=%USERPROFILE%\Documents\Intel\OpenVINO\venv_openvino

IF /I "%1%" EQU "" (
    set postfix=
) ELSE (
    IF /I "%1%" EQU "venv" (
        set postfix=
        set USE_VENV="true"
    ) ELSE (
    IF /I "%1%" EQU "caffe" (
        set postfix=_caffe
    ) ELSE (
    IF /I "%1%" EQU "tf" (
        set postfix=_tf
    ) ELSE (
    IF /I "%1%" EQU "tf2" (
        set postfix=_tf2
    ) ELSE (
    IF /I "%1%" EQU "mxnet" (
        set postfix=_mxnet
    ) ELSE (
    IF /I "%1%" EQU "kaldi" (
        set postfix=_kaldi
    ) ELSE (
    IF /I "%1%" EQU "onnx" (
        set postfix=_onnx
    ) ELSE (
           echo Unsupported framework
           goto error
       )
      )
     )
    )
   )
  )
 )
)

IF /I "%2%" EQU "venv" (
    set USE_VENV="true"
)

IF %USE_VENV% == "true" (
    python -m venv "%VENV_DIR%"
    call "%VENV_DIR%\Scripts\activate.bat"
)

python -m pip install -U pip
python -m pip install -r "%ROOT_DIR%..\requirements%postfix%.txt"

:: Chek MO version
set python_command='python "%ROOT_DIR%..\openvino\tools\mo\utils\extract_release_version.py"'
FOR /F "delims=" %%i IN (%python_command%) DO set mo_release_version=%%i
IF "%mo_release_version%" == "None.None" (
    set mo_is_custom="true"
) ELSE (
    set mo_is_custom="false"
)

:: Check if existing IE Python bindings satisfy requirements
set errorlevel=
python "%ROOT_DIR%..\openvino\tools\mo\utils\find_ie_version.py"
IF %errorlevel% EQU 0 goto ie_search_end

:: Check if OV already installed via pip
set errorlevel=
python -m pip show openvino
IF %errorlevel% EQU 0 (
    IF %mo_is_custom% == "true" (
        echo [ WARNING ] OpenVINO ^(TM^) Toolkit version installed in pip is incompatible with the Model Optimizer
        echo [ WARNING ] For the custom Model Optimizer version consider building Inference Engine Python API from sources ^(preferable^) or install the highest OpenVINO ^(TM^) toolkit version using "pip install openvino"
        goto ie_search_end
    )
    IF %mo_is_custom% == "false" (
        echo [ WARNING ] OpenVINO ^(TM^) Toolkit version installed in pip is incompatible with the Model Optimizer
        echo [ WARNING ] For the release Model Optimizer version which is %mo_release_version% please install OpenVINO ^(TM^) toolkit using pip install openvino==%mo_release_version% or build Inference Engine Python API from sources
        goto ie_search_end
    )
)

echo [ WARNING ] Could not find the Inference Engine Python API. Installing OpenVINO ^(TM^) toolkit using pip

IF %mo_is_custom% == "true" (
    echo [ WARNING ] Detected a custom Model Optimizer version
    echo [ WARNING ] The desired version of the Inference Engine can be installed only for the release Model Optimizer version
    echo [ WARNING ] The highest OpenVINO ^(TM^) toolkit version will be installed ^(may be incompatible with current Model Optimizer version^)
    echo [ WARNING ] It is recommended to build the Inference Engine from sources even if the current installation is successful
    goto install_last_ov
)

set errorlevel=
python -m pip install openvino==%mo_release_version%
IF %errorlevel% NEQ 0 (
    echo [ WARNING ] Could not find the OpenVINO ^(TM^) toolkit version %mo_release_version% in pip
    echo [ WARNING ] The highest OpenVINO ^(TM^) toolkit version will be installed ^(may be incompatible with current Model Optimizer version^)
    echo [ WARNING ] It is recommended to build the Inference Engine from sources even if the current installation is successful
    goto install_last_ov
)

set errorlevel=
python "%ROOT_DIR%..\openvino\tools\mo\utils\find_ie_version.py"
IF %errorlevel% EQU 0 goto ie_search_end

echo [ WARNING ] The installed OpenVINO ^(TM^) toolkit version %mo_release_version% does not work as expected. Uninstalling...
python -m pip uninstall -y openvino
echo [ WARNING ] Consider building the Inference Engine Python API from sources
goto ie_search_end

:install_last_ov
set errorlevel=
python -m pip install openvino
IF %errorlevel% NEQ 0 (
    echo [ WARNING ] Could not find OpenVINO ^(TM^) toolkit version available in pip for installation
    echo [ WARNING ] Consider building the Inference Engine Python API from sources
    goto ie_search_end
)

set errorlevel=
python "%ROOT_DIR%..\openvino\tools\mo\utils\find_ie_version.py"
IF %errorlevel% EQU 0 goto ie_search_end

echo [ WARNING ] The installed highest OpenVINO ^(TM^) toolkit version doesn't work as expected. Uninstalling...
python -m pip uninstall -y openvino
echo [ WARNING ] Consider building the Inference Engine Python API from sources
goto ie_search_end

:ie_search_end

IF %USE_VENV% == "true" (
    echo.
    echo Before running the Model Optimizer, please activate virtualenv environment by running "%VENV_DIR%\Scripts\activate.bat"
) ELSE (
    echo.
    echo [ WARNING ] All Model Optimizer dependencies are installed globally.
    echo [ WARNING ] If you want to keep Model Optimizer in separate sandbox
    echo [ WARNING ] run install_prerequisites.bat "{caffe|tf|tf2|mxnet|kaldi|onnx}" venv
)

goto:eof

:error
echo.


