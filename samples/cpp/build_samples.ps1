# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Arguments parsing
param (
    [Alias("b")]
    [string]$BuildDirectory = "",

    [Alias("i")]
    [string]$InstallDirectory = "",

    [Alias("h")]
    [switch]$Help
)

$ErrorActionPreference = "Stop"

$SourceDirectory = Split-Path $MyInvocation.MyCommand.Path
$SamplesType = (Get-Item $SourceDirectory).Name
$DefaultBuildDirectory = "$Env:USERPROFILE/Documents/Intel/OpenVINO/openvino_${SamplesType}_samples_build"

if ($Help) {
    Write-Host "
        Build OpenVINO Runtime samples
        Options:
            -h/-Help                         Print the help message and exit
            -b/-BuildDirectory               Specify the samples build directory. Default is '$DefaultBuildDirectory'
            -i/-InstallDirectory             Specify the samples install directory
    "
    exit 0
}

$BuildDirectory = if ($BuildDirectory) {$BuildDirectory} else {$DefaultBuildDirectory}

if (-not $Env:INTEL_OPENVINO_DIR) {
    $SetupVars = Join-Path $SourceDirectory "../../setupvars.ps1"
    if (Test-Path $SetupVars) {
        . $SetupVars
    }
    else
    {
        Write-Host "
            Failed to set the environment variables automatically. To fix, run the following command:
            <INTEL_OPENVINO_DIR>/setupvars.ps1
            where <INTEL_OPENVINO_DIR> is the OpenVINO installation directory
        "
        exit 1
    }
}

Set-Location -Path $SourceDirectory
New-Item -Path $BuildDirectory -ItemType Directory -Force
Set-Location $BuildDirectory
cmake -DCMAKE_DISABLE_FIND_PACKAGE_PkgConfig=ON $SourceDirectory

Write-Host "Building command: cmake --build . --config Release --parallel"
cmake --build . --config Release --parallel

if ($InstallDirectory) {
    cmake -DCMAKE_INSTALL_PREFIX="$InstallDirectory" -DCOMPONENT=samples_bin -P "$BuildDirectory/cmake_install.cmake"
    Write-Host "Samples are built and installed into $InstallDirectory"
}
else
{
    Write-Host "Samples are built in $BuildDirectory"
}
