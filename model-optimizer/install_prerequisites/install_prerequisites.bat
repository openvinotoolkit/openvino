@echo off
:: Copyright (C) 2018-2021 Intel Corporation
::
::  Licensed under the Apache License, Version 2.0 (the "License");
::  you may not use this file except in compliance with the License.
::  You may obtain a copy of the License at
::
::       http://www.apache.org/licenses/LICENSE-2.0
::
::  Unless required by applicable law or agreed to in writing, software
::  distributed under the License is distributed on an "AS IS" BASIS,
::  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
::  See the License for the specific language governing permissions and
::  limitations under the License.

:: Check if Python is installed
setlocal

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


IF /I "%1%" EQU "" (
    set postfix=
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

pip3 install --user -r ..\requirements%postfix%.txt

:: Chek MO version
python %~dp0..\mo\utils\extract_release_version.py
set python_command='python %~dp0..\mo\utils\extract_release_version.py'
FOR /F "delims=" %%i IN (%python_command%) DO set mo_release_version=%%i
IF "%mo_release_version%" == "None.None" (
    set mo_is_custom="true"
) ELSE (
    set mo_is_custom="false"
)

:: Check if existing IE Python bindings satisfy requirements
set errorlevel=
python %~dp0..\mo\utils\find_ie_version.py
IF %errorlevel% EQU 0 goto ie_search_end

:: Check if OV already installed via pip
set errorlevel=
pip3 show openvino
IF %errorlevel% EQU 0 (
    IF %mo_is_custom% == "true" (
        echo [ WARNING ] Existing "OpenVINO (TM) Toolkit" pip version is incompatible with Model Optimizer
        echo [ WARNING ] For not release Model Optimizer version please build Inference Engine Python API from sources ^(preferable^) or install latest "OpenVINO (TM) Toolkit" version using pip install openvino ^(may be incompatible^)
        goto ie_search_end
    )
    IF %mo_is_custom% == "false" (
        echo [ WARNING ] Existing "OpenVINO (TM) Toolkit" pip version is incompatible with Model Optimizer
        echo [ WARNING ] For release Model Optimizer version which is %mo_release_version% please install "OpenVINO (TM) Toolkit" using pip install openvino==%mo_release_version% or build Inference Engine Python API from sources
        goto ie_search_end
    )
)

echo [ WARNING ] No available Inference Engine Python API was found. Trying to install "OpenVINO (TM) Toolkit" using pip

IF %mo_is_custom% == "true" (
    echo [ WARNING ] Custom Model Optimizer version detected
    echo [ WARNING ] The desired version of Inference Engine can be installed only for release Model Optimizer version
    echo [ WARNING ] The latest "OpenVINO (TM) Toolkit" version will be installed ^(may be incompatible with current Model Optimizer version^)
    echo [ WARNING ] It is recommended to build Inference Engine from sources even if installation will be successful
    goto install_last_ov
)

set errorlevel=
pip3 install openvino==%mo_release_version%
IF %errorlevel% NEQ 0 (
    echo [ WARNING ] Can not find "OpenVINO (TM) Toolkit" version %mo_release_version% in pip
    echo [ WARNING ] But the latest "OpenVINO (TM) Toolkit" version will be installed ^(may be incompatible with current Model Optimizer version^)
    echo [ WARNING ] It is recommended to build Inference Engine from sources even if installation will be successful
    goto install_last_ov
)

set errorlevel=
python %~dp0..\mo\utils\find_ie_version.py
IF %errorlevel% EQU 0 goto ie_search_end

echo [ WARNING ] Installed "OpenVINO (TM) Toolkit" version %mo_release_version% doesn't work as expected...Uninstalling...
pip3 uninstall -y openvino
echo [ WARNING ] Please consider to build Inference Engine Python API from sources
goto ie_search_end

:install_last_ov
set errorlevel=
pip3 install openvino
IF %errorlevel% NEQ 0 (
    echo [ WARNING ] No "OpenVINO (TM) Toolkit" version is available in pip for installation
    echo [ WARNING ] Please consider to build Inference Engine Python API from sources
    goto ie_search_end
)

set errorlevel=
python %~dp0..\mo\utils\find_ie_version.py
IF %errorlevel% EQU 0 goto ie_search_end

echo [ WARNING ] Installed latest "OpenVINO (TM) Toolkit" version doesn't work as expected...Uninstalling...
pip3 uninstall -y openvino
echo [ WARNING ] Please consider to build Inference Engine Python API from sources.
goto ie_search_end

:ie_search_end

echo *****************************************************************************************
echo Optional: To speed up model conversion process, install protobuf-*.egg located in the
echo "model-optimizer\install_prerequisites" folder or building protobuf library from sources.
echo For more information please refer to Model Optimizer FAQ, question #80.

goto:eof

:error
echo.


