# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Run from an x64 VS2022 Developer PowerShell. See docs/dev/test_coverage.md.
param(
    [string]$WorkDir = "$PSScriptRoot/../build-windows-npu-coverage",
    [ValidateRange(1, 256)][int]$Jobs = 8
)

$ErrorActionPreference = 'Stop'
function Invoke-Checked {
    param([string]$Command, [string[]]$Arguments)
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Command exited with code $LASTEXITCODE" }
}

if ($env:VSCMD_ARG_TGT_ARCH -ne 'x64' -or $env:VisualStudioVersion -notlike '17.*') {
    throw 'Use an x64 VS2022 Developer PowerShell (MSVC v143; CI uses toolset 14.42).'
}
foreach ($tool in @('git', 'cmake', 'ninja', 'cl', 'python')) {
    Get-Command $tool -ErrorAction Stop | Out-Null
}
$source = (Resolve-Path "$PSScriptRoot/..").Path
$WorkDir = [IO.Path]::GetFullPath($WorkDir)
$build = Join-Path $WorkDir 'build'
$package = Join-Path $WorkDir 'package'
if (Test-Path $package) {
    throw "Package already exists: $package. Move it aside or use another -WorkDir."
}

Invoke-Checked git @('-C', $source, 'submodule', 'update', '--init', '--recursive')
$options = @(
    '-G', 'Ninja Multi-Config', '-S', $source, '-B', $build,
    '-DCMAKE_CONFIGURATION_TYPES=Release',
    '-DENABLE_COVERAGE=ON', '-DENABLE_PDB_IN_RELEASE=ON',
    '-DENABLE_TESTS=ON', '-DENABLE_FUNCTIONAL_TESTS=ON',
    '-DENABLE_INTEL_NPU=ON', '-DENABLE_INTEL_CPU=ON', '-DENABLE_INTEL_GPU=OFF',
    '-DENABLE_PYTHON=OFF', '-DENABLE_WHEEL=OFF', '-DENABLE_JS=OFF', '-DENABLE_SAMPLES=OFF',
    '-DCMAKE_COMPILE_WARNING_AS_ERROR=ON', '-DENABLE_STRICT_DEPENDENCIES=OFF',
    '-DCMAKE_DISABLE_FIND_PACKAGE_PkgConfig=ON', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'
)
# Keep IR support, but avoid frontend model-generation dependencies in this C++ experiment.
foreach ($frontend in @('ONNX', 'PADDLE', 'PYTORCH', 'JAX', 'TF', 'TF_LITE', 'GGUF')) {
    $options += "-DENABLE_OV_${frontend}_FRONTEND=OFF"
}
Invoke-Checked cmake $options
$env:CMAKE_BUILD_PARALLEL_LEVEL = "$Jobs"
Invoke-Checked cmake @('--build', $build, '--config', 'Release', '--parallel', "$Jobs")
Invoke-Checked cmake @('--install', $build, '--config', 'Release', '--prefix', $package)
foreach ($component in @('tests', 'pdb')) {
    Invoke-Checked cmake @('--install', $build, '--config', 'Release', '--prefix', $package,
        '--component', $component)
}
foreach ($name in @('ov_npu_unit_tests.exe', 'openvino_intel_npu_plugin.dll')) {
    $modules = @(Get-ChildItem $package -Recurse -File -Filter $name)
    if ($modules.Count -ne 1 -or
        !(Test-Path ([IO.Path]::ChangeExtension($modules[0].FullName, '.pdb')))) {
        throw "Expected one $name with an adjacent PDB in $package"
    }
}
Push-Location $package
try {
    Invoke-Checked cmake @('-E', 'tar', 'cf', '../openvino-npu-coverage.zip', '--format=zip', '.')
} finally {
    Pop-Location
}
Write-Host "Package: $package"
Write-Host "Archive: $WorkDir/openvino-npu-coverage.zip"
