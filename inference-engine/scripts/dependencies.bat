@echo off
:: Copyright (C) 2018-2020 Intel Corporation
:: SPDX-License-Identifier: Apache-2.0

setlocal enabledelayedexpansion

for /f "delims=" %%x in (dependencies_64.txt) do (set "%%x")

for %%A in ("%MKL%") do set MKL_FILENAME=%%~nxA
for %%A in ("%OMP%") do set OMP_FILENAME=%%~nxA
for %%A in ("%MYRIAD%") do set MYRIAD_FILENAME=%%~nxA
for %%A in ("%GNA%") do set GNA_FILENAME=%%~nxA
for %%A in ("%OPENCV%") do set OPENCV_FILENAME=%%~nxA
for %%A in ("%HDDL%") do set HDDL_FILENAME=%%~nxA
for %%A in ("%VPU_FIRMWARE_MA2X8X%") do set VPU_FIRMWARE_MA2X8X_FILENAME=%%~nxA
for %%A in ("%TBB%") do set TBB_FILENAME=%%~nxA

call :DownloadFile MKL %MKL%
call :DownloadFile OMP %OMP%
call :DownloadFile MYRIAD %MYRIAD%
call :DownloadFile GNA %GNA%
call :DownloadFile OPENCV %OPENCV%
call :DownloadFile HDDL %HDDL%
call :DownloadFile VPU_FIRMWARE_MA2X8X %VPU_FIRMWARE_MA2X8X%
call :DownloadFile TBB %TBB%

for /f "delims=" %%x in (ld_library_rpath_64.txt) do (set "%%x")

set PATH=%DL_SDK_TEMP%\test_dependencies\MYRIAD\%MYRIAD_FILENAME%%MYRIAD%;%PATH%
set PATH=%DL_SDK_TEMP%\test_dependencies\MKL\%MKL_FILENAME%%MKL%;%PATH%
set PATH=%DL_SDK_TEMP%\test_dependencies\OMP\%OMP_FILENAME%%OMP%;%PATH%
set PATH=%DL_SDK_TEMP%\test_dependencies\GNA\%GNA_FILENAME%%GNA%;%PATH%
set PATH=%DL_SDK_TEMP%\test_dependencies\OPENCV\%OPENCV_FILENAME%%OPENCV%;%PATH%
set PATH=%DL_SDK_TEMP%\test_dependencies\TBB\%TBB_FILENAME%%TBB%;%PATH%

set PATH=%DL_SDK_TEMP%\test_dependencies\MYRIAD\%MYRIAD_FILENAME%%MYRIAD%;%PATH%

if not "%MYRIAD%"=="" (
	if exist "%DL_SDK_TEMP%\test_dependencies\MYRIAD\%MYRIAD_FILENAME%%MYRIAD%\mvnc" (
		echo xcopy.exe "%DL_SDK_TEMP%\test_dependencies\MYRIAD\%MYRIAD_FILENAME%%MYRIAD%" intel64  /S /I /Y /R
		xcopy.exe "%DL_SDK_TEMP%\test_dependencies\MYRIAD\%MYRIAD_FILENAME%%MYRIAD%" intel64  /S /I /Y /R
	)

	if exist "%DL_SDK_TEMP%\test_dependencies\MYRIAD\%MYRIAD_FILENAME%%MYRIAD%\..\bin\mvnc" (
		echo xcopy.exe "%DL_SDK_TEMP%\test_dependencies\MYRIAD\%MYRIAD_FILENAME%%MYRIAD%\..\bin\*" intel64  /S /I /Y /R
		xcopy.exe "%DL_SDK_TEMP%\test_dependencies\MYRIAD\%MYRIAD_FILENAME%%MYRIAD%\..\bin\*" intel64  /S /I /Y /R
	)
)

if not "%VPU_FIRMWARE_MA2X8X%"=="" (
	if exist "%DL_SDK_TEMP%\test_dependencies\VPU\%VPU_FIRMWARE_MA2X8X_FILENAME%" (
		echo xcopy.exe "%DL_SDK_TEMP%\test_dependencies\VPU\%VPU_FIRMWARE_MA2X8X_FILENAME%\*" intel64  /S /I /Y /R
		xcopy.exe "%DL_SDK_TEMP%\test_dependencies\VPU\%VPU_FIRMWARE_MA2X8X_FILENAME%\*" intel64  /S /I /Y /R
	)
)

set PATH=%DL_SDK_TEMP%\test_dependencies\HDDL\%HDDL_FILENAME%%HDDL%\..\bin;%PATH%

if not "%HDDL%"=="" (
	set HDDL_INSTALL_DIR=%DL_SDK_TEMP%\test_dependencies\HDDL\%HDDL_FILENAME%%HDDL%\..
	if exist "%DL_SDK_TEMP%\test_dependencies\VPU\%VPU_FIRMWARE_MA2X8X_FILENAME%" (
		echo xcopy.exe "%DL_SDK_TEMP%\test_dependencies\VPU\%VPU_FIRMWARE_MA2X8X_FILENAME%\*" %HDDL_INSTALL_DIR%\lib  /S /I /Y /R
		xcopy.exe "%DL_SDK_TEMP%\test_dependencies\VPU\%VPU_FIRMWARE_MA2X8X_FILENAME%\*" "%HDDL_INSTALL_DIR%\lib"  /S /I /Y /R
	)
)

echo PATH=%PATH%

endlocal & set PATH=%PATH%

exit /B %ERRORLEVEL%

:DownloadFile
set DEPENDENCY=%~1
set DEPENDENCY_URL=%~2
set DEPENDENCY_FILE=%~nx2
set DEPENDENCY_EXT=%~x2

if not "%DEPENDENCY_URL%"=="" (
	if not exist "%DL_SDK_TEMP%\test_dependencies\%DEPENDENCY%\%DEPENDENCY_FILE%" (
		mkdir "%DL_SDK_TEMP%\test_dependencies\%DEPENDENCY%\%DEPENDENCY_FILE%"
		for /L %%a in (1,1,10) do (
			powershell -command "iwr -outf '%DL_SDK_TEMP%\test_dependencies\%DEPENDENCY%\_%DEPENDENCY_FILE%' %DEPENDENCY_URL%"
			call "C:\Program Files\7-Zip\7z.exe" x -y %DL_SDK_TEMP%\test_dependencies\%DEPENDENCY%\_%DEPENDENCY_FILE% -o%DL_SDK_TEMP%\test_dependencies\%DEPENDENCY%\%DEPENDENCY_FILE%
			if !ERRORLEVEL! equ 0 goto :DownloadFileContinue
			timeout /T 15
		)
	)
)
goto:eof

:DownloadFileContinue
if "%DEPENDENCY_EXT%" == ".txz" call "C:\Program Files\7-Zip\7z.exe" x -y %DL_SDK_TEMP%\test_dependencies\%DEPENDENCY%\%DEPENDENCY_FILE%\_%DEPENDENCY_FILE:txz=tar% -o%DL_SDK_TEMP%\test_dependencies\%DEPENDENCY%\%DEPENDENCY_FILE%
del "%DL_SDK_TEMP%\test_dependencies\%DEPENDENCY%\_%DEPENDENCY_FILE%" /F /Q
goto:eof
