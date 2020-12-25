@echo off

:: Copyright (c) 2018-2020 Intel Corporation
::
:: Licensed under the Apache License, Version 2.0 (the "License");
:: you may not use this file except in compliance with the License.
:: You may obtain a copy of the License at
::
::      http://www.apache.org/licenses/LICENSE-2.0
::
:: Unless required by applicable law or agreed to in writing, software
:: distributed under the License is distributed on an "AS IS" BASIS,
:: WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
:: See the License for the specific language governing permissions and
:: limitations under the License.

set ROOT=%~dp0
call :GetFullPath "%ROOT%\.." ROOT
set SCRIPT_NAME=%~nx0

set "INTEL_OPENVINO_DIR=%ROOT%"
set "INTEL_CVSDK_DIR=%INTEL_OPENVINO_DIR%"

:: OpenCV
if exist "%INTEL_OPENVINO_DIR%\opencv\setupvars.bat" (
call "%INTEL_OPENVINO_DIR%\opencv\setupvars.bat"
) else (
set "OpenCV_DIR=%INTEL_OPENVINO_DIR%\opencv\x64\vc14\lib"
set "PATH=%INTEL_OPENVINO_DIR%\opencv\x64\vc14\bin;%PATH%"
)

:: Model Optimizer
if exist %INTEL_OPENVINO_DIR%\deployment_tools\model_optimizer (
set PYTHONPATH=%INTEL_OPENVINO_DIR%\deployment_tools\model_optimizer;%PYTHONPATH%
set "PATH=%INTEL_OPENVINO_DIR%\deployment_tools\model_optimizer;%PATH%"
)

:: Inference Engine
set "InferenceEngine_DIR=%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\share"
set "HDDL_INSTALL_DIR=%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\external\hddl"
set "OPENMP_DIR=%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\external\omp\lib"
set "GNA_DIR=%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\external\gna\lib"

set "PATH=%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\bin\intel64\Release;%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\bin\intel64\Debug;%HDDL_INSTALL_DIR%\bin;%OPENMP_DIR%;%GNA_DIR%;%PATH%"
if exist %INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\bin\intel64\arch_descriptions (
set ARCH_ROOT_DIR=%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\bin\intel64\arch_descriptions
)
if exist %INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\bin\intel64\arch_descriptions (
set ARCH_ROOT_DIR=%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\bin\intel64\arch_descriptions
)

:: TBB
if exist %INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\external\tbb (
set "PATH=%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\external\tbb\bin;%PATH%"
set "TBB_DIR=%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\external\tbb\cmake"
)

:: nGraph
if exist %INTEL_OPENVINO_DIR%\deployment_tools\ngraph (
set "PATH=%INTEL_OPENVINO_DIR%\deployment_tools\ngraph\lib;%PATH%"
set "ngraph_DIR=%INTEL_OPENVINO_DIR%\deployment_tools\ngraph\cmake"
)

:: Check if Python is installed
python --version 2>NUL
if errorlevel 1 (
   echo Error^: Python is not installed. Please install one of Python 3.6 - 3.8 ^(64-bit^) from https://www.python.org/downloads/
   exit /B 1
)

:: Check Python version
for /F "tokens=* USEBACKQ" %%F IN (`python --version 2^>^&1`) DO (
   set pyversion=%%F
)

for /F "tokens=1,2,3 delims=. " %%a in ("%pyversion%") do (
   set pyversion_major=%%b
   set pyversion_minor=%%c
)

if "%pyversion_major%" geq "3" (
   if "%pyversion_minor%" geq "6" (
      set check_pyversion=okay
   )
)

if not "%check_pyversion%"=="okay" (
   echo Unsupported Python version. Please install one of Python 3.6 - 3.8 ^(64-bit^) from https://www.python.org/downloads/
   exit /B 1
)

:: Check Python bitness
python -c "import sys; print(64 if sys.maxsize > 2**32 else 32)" 2 > NUL
if errorlevel 1 (
   echo Error^: Error during installed Python bitness detection
   exit /B 1
)

for /F "tokens=* USEBACKQ" %%F IN (`python -c "import sys; print(64 if sys.maxsize > 2**32 else 32)" 2^>^&1`) DO (
   set bitness=%%F
)

if not "%bitness%"=="64" (
   echo Unsupported Python bitness. Please install one of Python 3.6 - 3.8 ^(64-bit^) from https://www.python.org/downloads/
   exit /B 1
)

set PYTHONPATH=%INTEL_OPENVINO_DIR%\python\python%pyversion_major%.%pyversion_minor%;%INTEL_OPENVINO_DIR%\python\python3;%PYTHONPATH%

if exist %INTEL_OPENVINO_DIR%\deployment_tools\open_model_zoo\tools\accuracy_checker (
    set PYTHONPATH=%INTEL_OPENVINO_DIR%\deployment_tools\open_model_zoo\tools\accuracy_checker;%PYTHONPATH%
)

if exist %INTEL_OPENVINO_DIR%\deployment_tools\tools\post_training_optimization_toolkit (
    set PYTHONPATH=%INTEL_OPENVINO_DIR%\deployment_tools\tools\post_training_optimization_toolkit;%PYTHONPATH%
)

echo [setupvars.bat] OpenVINO environment initialized

exit /B 0

:GetFullPath
SET %2=%~f1

GOTO :EOF
