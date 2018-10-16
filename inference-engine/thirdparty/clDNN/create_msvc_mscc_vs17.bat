@REM Copyright (c) 2016 Intel Corporation

@REM Licensed under the Apache License, Version 2.0 (the "License");
@REM you may not use this file except in compliance with the License.
@REM You may obtain a copy of the License at

@REM      http://www.apache.org/licenses/LICENSE-2.0

@REM Unless required by applicable law or agreed to in writing, software
@REM distributed under the License is distributed on an "AS IS" BASIS,
@REM WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@REM See the License for the specific language governing permissions and
@REM limitations under the License.


@setlocal
@echo off

set "ROOT_DIR=%~dp0"
set "BOOST_VERSION=1.64.0"
set "SOLUTION_TARGET32=Windows32"
set "SOLUTION_DIR32=%ROOT_DIR%\build\%SOLUTION_TARGET32%"

set "SOLUTION_TARGET64=Windows64"
set "SOLUTION_DIR64=%ROOT_DIR%\build\%SOLUTION_TARGET64%"

del %SOLUTION_DIR32%\CMakeCache.txt
del %SOLUTION_DIR64%\CMakeCache.txt
rmdir /S /Q %SOLUTION_DIR32%\codegen
rmdir /S /Q %SOLUTION_DIR64%\codegen

echo Creating Visual Studio 2017 (Win32) files in %SOLUTION_DIR32%... && ^
cd "%ROOT_DIR%" && cmake -E make_directory "%SOLUTION_DIR32%" && cd "%SOLUTION_DIR32%" && cmake -G "Visual Studio 15 2017" "-DCLDNN__ARCHITECTURE_TARGET=%SOLUTION_TARGET32%" "-DCLDNN__BOOST_VERSION=%BOOST_VERSION%" "%ROOT_DIR%"
echo Creating Visual Studio 2017 (x64) files in %SOLUTION_DIR64%... && ^
cd "%ROOT_DIR%" && cmake -E make_directory "%SOLUTION_DIR64%" && cd "%SOLUTION_DIR64%" && cmake -G "Visual Studio 15 2017 Win64" "-DCLDNN__ARCHITECTURE_TARGET=%SOLUTION_TARGET64%" "-DCLDNN__BOOST_VERSION=%BOOST_VERSION%" "%ROOT_DIR%"

echo Done.
pause
