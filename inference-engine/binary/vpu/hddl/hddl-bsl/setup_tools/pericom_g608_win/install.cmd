@echo off

if not defined PROCESSOR_ARCHITEW6432 (
    set url=https://www.diodes.com/assets/Uploads/disableacs64.zip
    set exe_name=disableacs64.exe
) else (
    set url=https://www.diodes.com/assets/Uploads/disableacs.zip
    set exe_name=disableacs.exe
)
    set input_program=%~dp0%exe_name%

set target_path=C:\pericom_g608.exe

if exist %input_program% (
    goto set_task
) else (
    goto suggest_download
)

:set_task
net session >nul 2>&1
if not %errorLevel% == 0 (
    echo Please run me with "Run as administrator"
    goto end
)

echo Copying from %input_program% to %target_path%
copy %input_program% %target_path%
echo Setting up Schedualed Task
schtasks /create /sc ONSTART /tn "HDDL Set Pericom G608" /tr %target_path% /it /ru System /f
goto end

:suggest_download
echo Please download %url%
echo Then unzip the file and put %exe_name% next to this script

:end
pause