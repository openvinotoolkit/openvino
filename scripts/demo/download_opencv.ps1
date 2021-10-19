# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Script parameters
param ([switch]$batch = $false)
if ($batch) { echo "Batch mode enabled ($batch)" } else { echo "Interactive mode, use -batch for batch mode"}


# Helper functions
Function Batch-Exit([int]$code = 0)
{
    if (!$batch) { Pause }
    Exit $code
}

# Elevate privileges if necessary
# Note: assuming we are in <openvino>/samples/scripts, archive should be unpacked in <openvino>
$openvino_root = [System.IO.Path]::GetFullPath("$PSScriptRoot\..\..")
$probe_file = "$openvino_root\dummy.txt"
try {
    [io.file]::OpenWrite($probe_file).close()
} catch {
    Write-Warning "Unable to write to $openvino_root"
    $identity = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
    if (!$identity.IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
        $opt = ""
        if ($batch) { $opt = $opt + "-batch" }
        Write-Warning "Launching with elevated privileges..."
        Start-Process `
            -FilePath powershell.exe `
            -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" $opt" `
            -Verb RunAs
        Batch-Exit
    } else {
        Write-Warning "Unable to write elevate privileges."
        Batch-Exit 1
    }
}
if([System.IO.File]::Exists($probe_file)) { rm "$probe_file" }

# Default download location
if(!$env:OPENVINO_OPENCV_DOWNLOAD_SERVER) {
    Write-Warning "OPENVINO_OPENCV_DOWNLOAD_SERVER has not been set, using default location."
    $url_root = "https://storage.openvinotoolkit.org/repositories/openvino/packages/master/opencv"
} else {
    $url_root = "$env:OPENVINO_OPENCV_DOWNLOAD_SERVER"
}

# Download archive
$url = "$url_root/windows.tgz"
$archive_file = "$env:TEMP\openvino_opencv_windows.tgz"
echo "* Download"
echo "  URL : $url"
echo "  DST : $archive_file"
try {
    Import-Module BitsTransfer
    Start-BitsTransfer -Source $url -Destination $archive_file -ErrorAction 'Stop'
} catch {
    Write-Warning "$_"
    Write-Warning "Download failed. Check the error message above for details."
    Batch-Exit 1
}

# Extra check
if(![System.IO.File]::Exists($archive_file)) {
    Write-Warning ("Download failed. File have not bi written to disk.")
    Batch-Exit 1
}

# Unpack tar to OpenVINO root
Start-Process `
    -FilePath "tar" `
    -ArgumentList "-x", "-f", "$archive_file" `
    -WorkingDirectory "$openvino_root" `
    -Wait `
    -NoNewWindow

# Remove temporary file
echo "* Cleanup"
echo "  FILE : $archive_file"
rm "$archive_file"
Batch-Exit
