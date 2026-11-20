#requires -Version 5.1
#requires -RunAsAdministrator
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

param(
    [string]$DownloadDir = "$PSScriptRoot/../build-windows-npu-coverage/tools",
    [string]$InstallDir = "$env:ProgramW6432/OpenCppCoverage"
)

$ErrorActionPreference = 'Stop'
if ($env:OS -ne 'Windows_NT' -or ![Environment]::Is64BitOperatingSystem) {
    throw 'This installer requires 64-bit Windows.'
}
$DownloadDir = [IO.Path]::GetFullPath($DownloadDir)
$InstallDir = [IO.Path]::GetFullPath($InstallDir)
$version = '0.9.9.0'
$expectedHash = '2295a733da39412c61e4f478677519dd0bb1893d88313ce56b468c9e50517888'
$installer = Join-Path $DownloadDir "OpenCppCoverageSetup-x64-$version.exe"
$log = Join-Path $DownloadDir "OpenCppCoverage-install-$(Get-Date -Format 'yyyyMMdd-HHmmss-fff').log"
$url = "https://github.com/OpenCppCoverage/OpenCppCoverage/releases/download/release-$version/OpenCppCoverageSetup-x64-$version.exe"
New-Item -ItemType Directory -Path $DownloadDir -Force | Out-Null

$savedProtocol = [Net.ServicePointManager]::SecurityProtocol
try {
    [Net.ServicePointManager]::SecurityProtocol = $savedProtocol -bor [Net.SecurityProtocolType]::Tls12
    Write-Host "Downloading $url"
    Invoke-WebRequest -Uri $url -OutFile $installer -UseBasicParsing
} finally {
    [Net.ServicePointManager]::SecurityProtocol = $savedProtocol
}
if ((Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash -ne $expectedHash) {
    throw "Installer SHA256 mismatch. Not executing $installer."
}

Write-Host "Installing into $InstallDir. Log: $log"
$setup = Start-Process -FilePath $installer -ArgumentList @(
    '/VERYSILENT', '/SUPPRESSMSGBOXES', '/SP-', '/NORESTART', '/RESTARTEXITCODE=3010',
    '/NOCLOSEAPPLICATIONS', '/NORESTARTAPPLICATIONS', "/DIR=`"$InstallDir`"", "/LOG=`"$log`""
) -Wait -PassThru
if ($setup.ExitCode -eq 3010) {
    throw "Installation requires a restart; no restart was performed. Log: $log"
}
if ($setup.ExitCode -ne 0) {
    throw "OpenCppCoverage installer exited with code $($setup.ExitCode). Log: $log"
}
$tool = Join-Path $InstallDir 'OpenCppCoverage.exe'
if (!(Test-Path -LiteralPath $tool -PathType Leaf)) {
    throw "Installer did not produce $tool. Log: $log"
}
$check = Start-Process -FilePath $tool -ArgumentList '--help' -WorkingDirectory $DownloadDir -NoNewWindow -Wait -PassThru
if ($check.ExitCode -ne 0) {
    throw "OpenCppCoverage --help failed with code $($check.ExitCode). Check installed dependencies."
}
Write-Host "OpenCppCoverage is ready. Use -ToolPath '$tool' with run_windows_npu_coverage.ps1."
