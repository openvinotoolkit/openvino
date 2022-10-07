@echo off

for /f "tokens=2" %%i in ('"pip3 show paddlepaddle | findstr Location"') do (
	set dst_path=%dst_path%%%i
)

set paddle_file=%1
copy /y %paddle_file:/=\% %dst_path%\paddle\dataset\

exit
