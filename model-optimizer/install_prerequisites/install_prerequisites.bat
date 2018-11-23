@echo off
:: Copyright (c) 2018 Intel Corporation
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


pip install --user -r ..\requirements%postfix%.txt

echo *****************************************************************************************
echo Warning: please expect that Model Optimizer conversion might be slow.
echo You can boost conversion speed by installing protobuf-*.egg located in the
echo "model-optimizer\install_prerequisites" folder or building protobuf library from sources.
echo For more information please refer to Model Optimizer FAQ, question #80.

goto:eof

:error
echo.


