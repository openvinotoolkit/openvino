@echo off
:: Copyright (C) 2018-2020 Intel Corporation
:: SPDX-License-Identifier: Apache-2.0

for /f "delims=" %%x in (dependencies_64.txt) do (set "%%x")

for %%A in ("%MKL%") do set MKL_FILENAME=%%~nxA
for %%A in ("%OMP%") do set OMP_FILENAME=%%~nxA
for %%A in ("%MYRIAD%") do set MYRIAD_FILENAME=%%~nxA
for %%A in ("%GNA%") do set GNA_FILENAME=%%~nxA
for %%A in ("%OPENCV%") do set OPENCV_FILENAME=%%~nxA
for %%A in ("%MYRIAD%") do set MYRIAD_FILENAME=%%~nxA
for %%A in ("%HDDL%") do set HDDL_FILENAME=%%~nxA
for %%A in ("%VPU_FIRMWARE_MA2450%") do set VPU_FIRMWARE_MA2450_FILENAME=%%~nxA
for %%A in ("%VPU_FIRMWARE_MA2X8X%") do set VPU_FIRMWARE_MA2X8X_FILENAME=%%~nxA
for %%A in ("%TBB%") do set TBB_FILENAME=%%~nxA

if not "%MKL%"=="" (
	if not exist "%DL_SDK_TEMP%\test_dependencies\MKL\%MKL_FILENAME%" (
		mkdir "%DL_SDK_TEMP%\test_dependencies\MKL"
		powershell -command "iwr -outf '%DL_SDK_TEMP%\test_dependencies\MKL\_%MKL_FILENAME%' %MKL%"
		mkdir "%DL_SDK_TEMP%\test_dependencies\MKL\%MKL_FILENAME%"
		call "C:\Program Files\7-Zip\7z.exe" x -y %DL_SDK_TEMP%\test_dependencies\MKL\_%MKL_FILENAME% -o%DL_SDK_TEMP%\test_dependencies\MKL\%MKL_FILENAME%
		del "%DL_SDK_TEMP%\test_dependencies\MKL\_%MKL_FILENAME%" /F /Q
	)
)

if not "%OMP%"=="" (
	if not exist "%DL_SDK_TEMP%\test_dependencies\OMP\%OMP_FILENAME%" (
		mkdir "%DL_SDK_TEMP%\test_dependencies\OMP"
		powershell -command "iwr -outf '%DL_SDK_TEMP%\test_dependencies\OMP\_%OMP_FILENAME%' %OMP%"
		mkdir "%DL_SDK_TEMP%\test_dependencies\OMP\%OMP_FILENAME%"
		call "C:\Program Files\7-Zip\7z.exe" x -y %DL_SDK_TEMP%\test_dependencies\OMP\_%OMP_FILENAME% -o%DL_SDK_TEMP%\test_dependencies\OMP\%OMP_FILENAME%
		del "%DL_SDK_TEMP%\test_dependencies\OMP\_%OMP_FILENAME%" /F /Q
	)
)

if not "%MYRIAD%"=="" (
	if not exist "%DL_SDK_TEMP%\test_dependencies\MYRIAD\%MYRIAD_FILENAME%" (
		mkdir "%DL_SDK_TEMP%\test_dependencies\MYRIAD"
		powershell -command "iwr -outf '%DL_SDK_TEMP%\test_dependencies\MYRIAD\_%MYRIAD_FILENAME%' %MYRIAD%"
		mkdir "%DL_SDK_TEMP%\test_dependencies\MYRIAD\%MYRIAD_FILENAME%"
		call "C:\Program Files\7-Zip\7z.exe" x -y %DL_SDK_TEMP%\test_dependencies\MYRIAD\_%MYRIAD_FILENAME% -o%DL_SDK_TEMP%\test_dependencies\MYRIAD\%MYRIAD_FILENAME%
		del "%DL_SDK_TEMP%\test_dependencies\MYRIAD\_%MYRIAD_FILENAME%" /F /Q
	)
)

if not "%GNA%"=="" (
	if not exist "%DL_SDK_TEMP%\test_dependencies\GNA\%GNA_FILENAME%" (
		mkdir "%DL_SDK_TEMP%\test_dependencies\GNA"
		powershell -command "iwr -outf '%DL_SDK_TEMP%\test_dependencies\GNA\_%GNA_FILENAME%' %GNA%"
		mkdir "%DL_SDK_TEMP%\test_dependencies\GNA\%GNA_FILENAME%"
		call "C:\Program Files\7-Zip\7z.exe" x -y %DL_SDK_TEMP%\test_dependencies\GNA\_%GNA_FILENAME% -o%DL_SDK_TEMP%\test_dependencies\GNA\%GNA_FILENAME%
		del "%DL_SDK_TEMP%\test_dependencies\GNA\_%GNA_FILENAME%" /F /Q
	)
)

if not "%OPENCV%"=="" (
	if not exist "%DL_SDK_TEMP%\test_dependencies\OPENCV\%OPENCV_FILENAME%" (
		mkdir "%DL_SDK_TEMP%\test_dependencies\OPENCV"
		powershell -command "iwr -outf '%DL_SDK_TEMP%\test_dependencies\OPENCV\_%OPENCV_FILENAME%' %OPENCV%"
		mkdir "%DL_SDK_TEMP%\test_dependencies\OPENCV\%OPENCV_FILENAME%"
		call "C:\Program Files\7-Zip\7z.exe" x -y %DL_SDK_TEMP%\test_dependencies\OPENCV\_%OPENCV_FILENAME% -o%DL_SDK_TEMP%\test_dependencies\OPENCV\%OPENCV_FILENAME%
        call "C:\Program Files\7-Zip\7z.exe" x -y %DL_SDK_TEMP%\test_dependencies\OPENCV\%OPENCV_FILENAME%\_%OPENCV_FILENAME:txz=tar% -o%DL_SDK_TEMP%\test_dependencies\OPENCV\%OPENCV_FILENAME%
		del "%DL_SDK_TEMP%\test_dependencies\OPENCV\_%OPENCV_FILENAME%" /F /Q
	)
)

if not "%MYRIAD%"=="" (
	if not exist "%DL_SDK_TEMP%\test_dependencies\MYRIAD\%MYRIAD_FILENAME%" (
		mkdir "%DL_SDK_TEMP%\test_dependencies\MYRIAD"
		powershell -command "iwr -outf '%DL_SDK_TEMP%\test_dependencies\MYRIAD\_%MYRIAD_FILENAME%' %MYRIAD%"
		mkdir "%DL_SDK_TEMP%\test_dependencies\MYRIAD\%MYRIAD_FILENAME%"
		call "C:\Program Files\7-Zip\7z.exe" x -y %DL_SDK_TEMP%\test_dependencies\MYRIAD\_%MYRIAD_FILENAME% -o%DL_SDK_TEMP%\test_dependencies\MYRIAD\%MYRIAD_FILENAME%
		del "%DL_SDK_TEMP%\test_dependencies\MYRIAD\_%MYRIAD_FILENAME%" /F /Q
	)
)

if not "%HDDL%"=="" (
	if not exist "%DL_SDK_TEMP%\test_dependencies\HDDL\%HDDL_FILENAME%" (
		mkdir "%DL_SDK_TEMP%\test_dependencies\HDDL"
		powershell -command "iwr -outf '%DL_SDK_TEMP%\test_dependencies\HDDL\_%HDDL_FILENAME%' %HDDL%"
		mkdir "%DL_SDK_TEMP%\test_dependencies\HDDL\%HDDL_FILENAME%"
		call "C:\Program Files\7-Zip\7z.exe" x -y %DL_SDK_TEMP%\test_dependencies\HDDL\_%HDDL_FILENAME% -o%DL_SDK_TEMP%\test_dependencies\HDDL\%HDDL_FILENAME%
		del "%DL_SDK_TEMP%\test_dependencies\HDDL\_%HDDL_FILENAME%" /F /Q
	)
)

if not "%VPU_FIRMWARE_MA2450%"=="" (
	if not exist "%DL_SDK_TEMP%\test_dependencies\VPU\%VPU_FIRMWARE_MA2450_FILENAME%" (
		mkdir "%DL_SDK_TEMP%\test_dependencies\VPU"
		powershell -command "iwr -outf '%DL_SDK_TEMP%\test_dependencies\VPU\_%VPU_FIRMWARE_MA2450_FILENAME%' %VPU_FIRMWARE_MA2450%"
		mkdir "%DL_SDK_TEMP%\test_dependencies\VPU\%VPU_FIRMWARE_MA2450_FILENAME%"
		call "C:\Program Files\7-Zip\7z.exe" x -y %DL_SDK_TEMP%\test_dependencies\VPU\_%VPU_FIRMWARE_MA2450_FILENAME% -o%DL_SDK_TEMP%\test_dependencies\VPU\%VPU_FIRMWARE_MA2450_FILENAME%
		del "%DL_SDK_TEMP%\test_dependencies\VPU\_%VPU_FIRMWARE_MA2450_FILENAME%" /F /Q
	)
)

if not "%VPU_FIRMWARE_MA2X8X%"=="" (
	if not exist "%DL_SDK_TEMP%\test_dependencies\VPU\%VPU_FIRMWARE_MA2X8X_FILENAME%" (
		mkdir "%DL_SDK_TEMP%\test_dependencies\VPU"
		powershell -command "iwr -outf '%DL_SDK_TEMP%\test_dependencies\VPU\_%VPU_FIRMWARE_MA2X8X_FILENAME%' %VPU_FIRMWARE_MA2X8X%"
		mkdir "%DL_SDK_TEMP%\test_dependencies\VPU\%VPU_FIRMWARE_MA2X8X_FILENAME%"
		call "C:\Program Files\7-Zip\7z.exe" x -y %DL_SDK_TEMP%\test_dependencies\VPU\_%VPU_FIRMWARE_MA2X8X_FILENAME% -o%DL_SDK_TEMP%\test_dependencies\VPU\%VPU_FIRMWARE_MA2X8X_FILENAME%
		del "%DL_SDK_TEMP%\test_dependencies\VPU\_%VPU_FIRMWARE_MA2X8X_FILENAME%" /F /Q
	)
)

if not "%TBB%"=="" (
	if not exist "%DL_SDK_TEMP%\test_dependencies\TBB\%TBB_FILENAME%" (
		mkdir "%DL_SDK_TEMP%\test_dependencies\TBB"
		powershell -command "iwr -outf '%DL_SDK_TEMP%\test_dependencies\TBB\_%TBB_FILENAME%' %TBB%"
		mkdir "%DL_SDK_TEMP%\test_dependencies\TBB\%TBB_FILENAME%"
		call "C:\Program Files\7-Zip\7z.exe" x -y %DL_SDK_TEMP%\test_dependencies\TBB\_%TBB_FILENAME% -o%DL_SDK_TEMP%\test_dependencies\TBB\%TBB_FILENAME%
		del "%DL_SDK_TEMP%\test_dependencies\TBB\_%TBB_FILENAME%" /F /Q
	)
)

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

if not "%VPU_FIRMWARE_MA2450%"=="" (
	if exist "%DL_SDK_TEMP%\test_dependencies\VPU\%VPU_FIRMWARE_MA2450_FILENAME%" (
		echo xcopy.exe "%DL_SDK_TEMP%\test_dependencies\VPU\%VPU_FIRMWARE_MA2450_FILENAME%\*" intel64  /S /I /Y /R
		xcopy.exe "%DL_SDK_TEMP%\test_dependencies\VPU\%VPU_FIRMWARE_MA2450_FILENAME%\*" intel64  /S /I /Y /R
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
