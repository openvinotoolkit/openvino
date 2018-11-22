@echo off

:: Copyright (c) 2018 Intel Corporation
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


@setlocal
set "ROOT_DIR=%~dp0"

set "SOLUTION_DIR64=%USERPROFILE%\Documents\Intel\OpenVINO\inference_engine_samples_2017"
if exist "%SOLUTION_DIR64%" rd /s /q "%SOLUTION_DIR64%"
if "%InferenceEngine_DIR%"=="" set "InferenceEngine_DIR=%ROOT_DIR%\..\share"
if exist "%ROOT_DIR%\..\..\bin\setupvars.bat" call "%ROOT_DIR%\..\..\bin\setupvars.bat"
if exist "%ROOT_DIR%\..\..\..\bin\setupvars.bat" call "%ROOT_DIR%\..\..\..\bin\setupvars.bat"

echo Creating Visual Studio 2017 (x64) files in %SOLUTION_DIR64%... && ^
cd "%ROOT_DIR%" && cmake -E make_directory "%SOLUTION_DIR64%" && cd "%SOLUTION_DIR64%" && cmake -G "Visual Studio 15 2017 Win64" "%ROOT_DIR%"

echo Done.
pause