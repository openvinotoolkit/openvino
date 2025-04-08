@echo off

REM Copyright (C) 2018-2025 Intel Corporation
REM SPDX-License-Identifier: Apache-2.0

setlocal EnableDelayedExpansion

REM Invalid expressions
set "invalid_expressions=ov::op::v1::Add ov::op::v1::Divide ov::op::v1::Multiply ov::op::v1::Subtract ov::op::v1::Divide ov::Node ov::Input<ov::Node> ov::descriptor::Tensor <Type: 'undefined'> ov::Output<ov::Node const> ov::float16 ov::EncryptionCallbacks ov::streams::Num <Dimension: dynamic ov::pass::pattern::PatternSymbolValue <RTMap>"

REM Invalid identifiers
set "invalid_identifiers=<locals>"

REM Unresolved names
set "unresolved_names=InferRequestWrapper RemoteTensorWrapper capsule VASurfaceTensorWrapper _abc._abc_data openvino._ov_api.undefined_deprecated  InputCutInfo ParamData"

REM Function to escape characters for regex
:escape_characters
set "escaped_error=%~1"
set "escaped_error=!escaped_error:\=\\!"
set "escaped_error=!escaped_error:/=\/!"
set "escaped_error=!escaped_error:$=\$!"
set "escaped_error=!escaped_error:*=\*!"
set "escaped_error=!escaped_error:.=\.!"
set "escaped_error=!escaped_error:^=\^!"
set "escaped_error=!escaped_error:|=\|!"
exit /b 0

REM Create regex pattern
set "invalid_expressions_regex="
for %%e in (%invalid_expressions%) do (
    call :escape_characters "%%e"
    set "invalid_expressions_regex=!invalid_expressions_regex!.*!escaped_error!.*|"
)
set "invalid_expressions_regex=%invalid_expressions_regex:~0,-1%"

set "invalid_identifiers_regex="
for %%e in (%invalid_identifiers%) do (
    call :escape_characters "%%e"
    set "invalid_identifiers_regex=!invalid_identifiers_regex!.*!escaped_error!.*|"
)
set "invalid_identifiers_regex=%invalid_identifiers_regex:~0,-1%"

set "unresolved_names_regex="
for %%e in (%unresolved_names%) do (
    call :escape_characters "%%e"
    set "unresolved_names_regex=!unresolved_names_regex!.*!escaped_error!.*|"
)
set "unresolved_names_regex=%unresolved_names_regex:~0,-1%"

REM Set the output directory
if "%~1"=="" (
    set "output_dir=%~dp0.."
) else (
    set "output_dir=%~1"
)

REM Generate stubs for C++ bindings
python -m pybind11_stubgen --output-dir "%output_dir%" --root-suffix "" --ignore-invalid-expressions "%invalid_expressions_regex%" --ignore-invalid-identifiers "%invalid_identifiers_regex%" --ignore-unresolved-names "%unresolved_names_regex%" --print-invalid-expressions-as-is --numpy-array-use-type-var --exit-code openvino

REM Check if the command was successful
if %errorlevel% neq 0 (
    echo Error: pybind11-stubgen failed.
    exit /b 1
)

REM Check if the stubs were actually generated
if exist "%output_dir%\openvino" (
    echo Stub files generated successfully.
) else (
    echo No stub files were generated.
    exit /b 1
)

REM Workaround for pybind11-stubgen issue where it doesn't import some modules for stubs generated from .py files
REM Ticket: 163225
set "pyi_file=%output_dir%\openvino\_ov_api.pyi"
if exist "%pyi_file%" (
    powershell -Command "(gc '%pyi_file%') -replace '(^.*$)', 'import typing`r`nimport pathlib`r`n$1' | Out-File -encoding ASCII '%pyi_file%'"
) else (
    echo File %pyi_file% not found.
    exit /b 1
)

REM Find all changed .pyi files
for /f "tokens=*" %%f in ('git diff --name-only ^| findstr /r "\.pyi$"') do (
    REM Process each changed .pyi file
    powershell -Command "(gc '%%f') -replace '<function _get_node_factory at 0x[0-9a-fA-F]+>', '<function _get_node_factory at memory_address>' | Out-File -encoding ASCII '%%f'"
    powershell -Command "(gc '%%f') -replace '__version__: str = ''[^'']*''', '__version__: str = ''version_string''' | Out-File -encoding ASCII '%%f'"
    powershell -Command "(gc '%%f') -replace '<function <lambda> at 0x[0-9a-fA-F]+>', '<function <lambda> at memory_address>' | Out-File -encoding ASCII '%%f'"
    powershell -Command "(gc '%%f') -replace ': \.\.\.', ': typing.Any' | Out-File -encoding ASCII '%%f'"
    powershell -Command "(gc '%%f') -replace 'pass: MatcherPass', 'matcher_pass: MatcherPass' | Out-File -encoding ASCII '%%f'"
    REM Sort consecutive import statements at the beginning of the file
    powershell -Command @"
    \$content = Get-Content '%%f'
    \$in_imports = \$false
    \$start = 0
    \$imports = @()

    foreach (\$line in \$content) {
        if (\$line -match '^from ' -or \$line -match '^import ') {
            if (-not \$in_imports) {
                \$start = \$true
            }
            \$in_imports = \$true
            \$imports += \$line
        } else {
            if (\$in_imports) {
                \$imports = \$imports | Sort-Object
                \$imports | Out-File -Append -Encoding ASCII '%%f'
                \$in_imports = \$false
                \$imports = @()
            }
            Add-Content -Path '%%f' -Value \$line
        }
    }

    if (\$in_imports) {
        \$imports = \$imports | Sort-Object
        \$imports | Out-File -Append -Encoding ASCII '%%f'
    }
    Add-Content -Path '%%f' -Value '# type: ignore'
"@
)

endlocal
