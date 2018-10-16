@echo off
rem ===============================================================================
rem Copyright 2018 Intel Corporation
rem
rem Licensed under the Apache License, Version 2.0 (the "License");
rem you may not use this file except in compliance with the License.
rem You may obtain a copy of the License at
rem
rem     http://www.apache.org/licenses/LICENSE-2.0
rem
rem Unless required by applicable law or agreed to in writing, software
rem distributed under the License is distributed on an "AS IS" BASIS,
rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
rem See the License for the specific language governing permissions and
rem limitations under the License.
rem ===============================================================================

rem req: PowerShell 3.0+
powershell.exe -command "if ($PSVersionTable.PSVersion.Major -ge 3) {exit 1} else {Write-Host \"The script requires PowerShell 3.0 or above (current version: $($PSVersionTable.PSVersion.Major).$($PSVersionTable.PSVersion.Minor))\"}" && goto Error_load

set MKLURLROOT=https://github.com/intel/mkl-dnn/releases/download/v0.15/
set MKLVERSION=2018.0.3.20180406

set MKLPACKAGE=mklml_win_%MKLVERSION%.zip

set MKLURL=%MKLURLROOT%%MKLPACKAGE%
if /i "%1"=="" (
	set DST=%~dp0..\external
) else (
	set DST=%1\..\external
)

if not exist %DST% mkdir %DST%

powershell.exe -command "if (Get-Command Invoke-WebRequest -errorAction SilentlyContinue){[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest %MKLURL% -OutFile %DST%\%MKLPACKAGE%} else {exit 1}" && goto Unpack || goto Error_load

:Unpack
powershell.exe -command "if (Get-Command Add-Type -errorAction SilentlyContinue) {Add-Type -Assembly \"System.IO.Compression.FileSystem\"; try { [IO.Compression.zipfile]::ExtractToDirectory(\"%DST%\%MKLPACKAGE%\", \"%DST%\")}catch{$_.exception ; exit 1}} else {exit 1}" && goto Exit || goto Error_unpack

:Error_load
echo prepare_mkl.bat : Error: Failed to load %MKLURL% to %DST%, try to load it manually
exit /B 1

:Error_unpack
echo prepare_mkl.bat : Error: Failed to unpack %DST%\%MKLPACKAGE% to %DST%, try unpack the archive manually
exit /B 1

:Exit
echo Downloaded and unpacked Intel(R) MKL small libraries to %DST%
exit /B 0