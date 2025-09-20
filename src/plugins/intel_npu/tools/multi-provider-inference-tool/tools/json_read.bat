echo off

echo "Use %%json_read%% to get formatted JSON"

setlocal

set data=
for /f "delims=" %%x in (%~1) do set "data=%data%%%x"

set data=%data:"=\"%
endlocal & set json_read=%data%

