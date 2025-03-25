# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Arguments parsing
param (
    [string]$python_version
)

$Env:INTEL_OPENVINO_DIR = Split-Path $MyInvocation.MyCommand.Path

$Env:OpenVINO_DIR = "$Env:INTEL_OPENVINO_DIR/runtime/cmake"
if (Test-Path -Path "$Env:OpenVINO_DIR/OpenVINOGenAIConfig.cmake")
{
    # If GenAI is installed, export it as well.
    $Env:OpenVINOGenAI_DIR = $Env:OpenVINO_DIR
}
$Env:OPENVINO_LIB_PATHS = "$Env:INTEL_OPENVINO_DIR/runtime/bin/intel64/Release;$Env:INTEL_OPENVINO_DIR/runtime/bin/intel64/Debug;$Env:OPENVINO_LIB_PATHS"

# TBB
if (Test-Path -Path "$Env:INTEL_OPENVINO_DIR/runtime/3rdparty/tbb")
{
    $prefix = ""
    if (Test-Path -Path "$Env:INTEL_OPENVINO_DIR/runtime/3rdparty/tbb/redist")
    {
        $prefix = "$Env:INTEL_OPENVINO_DIR/runtime/3rdparty/tbb/redist/intel64/vc14"
    }
    elseif (Test-Path -Path "$Env:INTEL_OPENVINO_DIR/runtime/3rdparty/tbb/bin/intel64/vc14")
    {
        $prefix = "$Env:INTEL_OPENVINO_DIR/runtime/3rdparty/tbb/bin/intel64/vc14"
    }
    elseif (Test-Path -Path "$Env:INTEL_OPENVINO_DIR/runtime/3rdparty/tbb/bin")
    {
        $prefix = "$Env:INTEL_OPENVINO_DIR/runtime/3rdparty/tbb/bin"
    }

    if ($prefix)
    {
        $Env:OPENVINO_LIB_PATHS = "$prefix;$Env:OPENVINO_LIB_PATHS"
    }

    if (Test-Path -Path "$Env:INTEL_OPENVINO_DIR/runtime/3rdparty/tbb/cmake")
    {
        $Env:TBB_DIR = "$Env:INTEL_OPENVINO_DIR/runtime/3rdparty/tbb/cmake"
    }
    elseif (Test-Path -Path "$Env:INTEL_OPENVINO_DIR/runtime/3rdparty/tbb/lib/cmake/TBB")
    {
        $Env:TBB_DIR = "$Env:INTEL_OPENVINO_DIR/runtime/3rdparty/tbb/lib/cmake/TBB"
    }
    elseif (Test-Path -Path "$Env:INTEL_OPENVINO_DIR/runtime/3rdparty/tbb/lib64/cmake/TBB")
    {
        $Env:TBB_DIR = "$Env:INTEL_OPENVINO_DIR/runtime/3rdparty/tbb/lib64/cmake/TBB"
    }
    elseif (Test-Path -Path "$Env:INTEL_OPENVINO_DIR/runtime/3rdparty/tbb/lib/cmake/tbb")
    {
        $Env:TBB_DIR = "$Env:INTEL_OPENVINO_DIR/runtime/3rdparty/tbb/lib/cmake/tbb"
    }
}

# Add libs directories to the PATH
$Env:PATH = "$Env:OPENVINO_LIB_PATHS;$Env:PATH"

Write-Host "[setupvars] OpenVINO environment initialized"

# Check if Python is installed
$PYTHON_VERSION_MAJOR = 3
$MIN_REQUIRED_PYTHON_VERSION_MINOR = 9
$MAX_SUPPORTED_PYTHON_VERSION_MINOR = 13

try
{
    # Should select the latest installed Python version as per https://docs.python.org/3/using/windows.html#getting-started
    (py --version) | Out-Null
}
catch
{
    Write-Host "Warning: Python is not installed. Please install one of Python $PYTHON_VERSION_MAJOR.$MIN_REQUIRED_PYTHON_VERSION_MINOR - $PYTHON_VERSION_MAJOR.$MAX_SUPPORTED_PYTHON_VERSION_MINOR (64-bit) from https://www.python.org/downloads/"
    # Python is not mandatory so we can safely exit with 0
    Exit 0
}

# Check Python version if user did not pass -python_version
if (-not $python_version)
{
    $installed_python_version_major = [int](py -c "import sys; print(f'{sys.version_info[0]}')")
    $installed_python_version_minor = [int](py -c "import sys; print(f'{sys.version_info[1]}')")
}
else
{
    [int]$installed_python_version_major, [int]$installed_python_version_minor = $python_version.Split('.')
}

if (-not ($PYTHON_VERSION_MAJOR -eq $installed_python_version_major -and $installed_python_version_minor -ge $MIN_REQUIRED_PYTHON_VERSION_MINOR -and $installed_python_version_minor -le $MAX_SUPPORTED_PYTHON_VERSION_MINOR))
{
    Write-Host "Warning: Unsupported Python version $installed_python_version_major.$installed_python_version_minor. Please install one of Python $PYTHON_VERSION_MAJOR.$MIN_REQUIRED_PYTHON_VERSION_MINOR - $PYTHON_VERSION_MAJOR.$MAX_SUPPORTED_PYTHON_VERSION_MINOR (64-bit) from https://www.python.org/downloads/"
    # Python is not mandatory so we can safely exit with 0
    Exit 0
}


# Check Python bitness
try
{
    $python_bitness = (py -c "import sys; print(64 if sys.maxsize > 2**32 else 32)")
}
catch
{
    Write-Host "Warning: Cannot determine installed Python bitness"
    # Python is not mandatory so we can safely exit with 0
    Exit 0
}

if ($python_bitness -ne "64")
{
    Write-Host "Warning: Unsupported Python bitness. Please install one of Python $PYTHON_VERSION_MAJOR.$MIN_REQUIRED_PYTHON_VERSION_MINOR - $PYTHON_VERSION_MAJOR.$MAX_SUPPORTED_PYTHON_VERSION_MINOR (64-bit) from https://www.python.org/downloads/"
    # Python is not mandatory so we can safely exit with 0
    Exit 0
}

$Env:PYTHONPATH = "$Env:INTEL_OPENVINO_DIR/python;$Env:INTEL_OPENVINO_DIR/python/python3;$Env:PYTHONPATH"

Write-Host "[setupvars] OpenVINO Python environment initialized"
