@echo off

:: Copyright (c) 2018-2019 Intel Corporation
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

where /q libmmd.dll || echo Warning: libmmd.dll couldn't be found in %%PATH%%. Please check if the redistributable package for Intel(R) C++ Compiler is installed and the library path is added to the PATH environment variable. System reboot can be required to update the system environment.

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
set "PATH=%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\external\tbb\bin;%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\bin\intel64\Release;%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\bin\intel64\Debug;%HDDL_INSTALL_DIR%\bin;%PATH%"
if exist "%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\bin\intel64\Release\arch_descriptions" (
set "ARCH_ROOT_DIR=%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\bin\intel64\Release\arch_descriptions"
)

:: nGraph
if exist %INTEL_OPENVINO_DIR%\deployment_tools\ngraph (
set "PATH=%INTEL_OPENVINO_DIR%\deployment_tools\ngraph\lib;%PATH%"
set "ngraph_DIR=%INTEL_OPENVINO_DIR%\deployment_tools\ngraph\cmake"
)

:: Check if Python is installed
python --version 2>NUL
if errorlevel 1 (
   echo Error^: Python is not installed. Please install Python 3.5. or 3.6  ^(64-bit^) from https://www.python.org/downloads/
   exit /B 1
)

:: Check Python version
for /F "tokens=* USEBACKQ" %%F IN (`python --version 2^>^&1`) DO (
   set version=%%F
)

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
   echo Unsupported Python version. Please install Python 3.5 or 3.6  ^(64-bit^) from https://www.python.org/downloads/
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
   echo Unsupported Python bitness. Please install Python 3.5 or 3.6  ^(64-bit^) from https://www.python.org/downloads/
   exit /B 1
)

set PYTHONPATH=%INTEL_OPENVINO_DIR%\python\python%Major%.%Minor%;%INTEL_OPENVINO_DIR%\python\python3;%PYTHONPATH%

if exist %INTEL_OPENVINO_DIR%\deployment_tools\open_model_zoo\tools\accuracy_checker (
    set PYTHONPATH=%INTEL_OPENVINO_DIR%\deployment_tools\open_model_zoo\tools\accuracy_checker;%PYTHONPATH%
)

echo [setupvars.bat] OpenVINO environment initialized

exit /B 0

:GetFullPath
SET %2=%~f1

GOTO :EOF
