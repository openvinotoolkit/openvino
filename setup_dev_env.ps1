# setup_dev_env.ps1
Write-Host ">>> Initializing OpenVINO Dev Environment..." -ForegroundColor Cyan

# 1. Define Root and Build Paths (ADJUST THESE IF YOUR BUILD FOLDER IS DIFFERENT)
$RepoRoot = Get-Location
$BuildDir = Join-Path $RepoRoot "build"
$ReleaseBin = Join-Path $RepoRoot "bin\intel64\Release"

# 2. Check if the build exists
if (-not (Test-Path $ReleaseBin)) {
    Write-Error "Build artifact path not found: $ReleaseBin. Did you build in Release mode?"
    exit 1
}

# 3. CRITICAL: Handle External Dependencies (TBB, etc.)
# If you used specific external deps or installed via vcpkg/conda, their DLLs must be here.
# Found TBB in temp/Windows_AMD64/tbb/bin
$TbbBin = Join-Path $RepoRoot "temp\Windows_AMD64\tbb\bin"

# Helper to avoid duplicate PATH entries
function Add-ToPath {
    param($PathToAdd)
    if ($Env:PATH -split ';' -notcontains $PathToAdd) {
        $Env:PATH = "$PathToAdd;$Env:PATH"
    }
}

Add-ToPath $ReleaseBin
Add-ToPath $TbbBin

# OPENVINO_LIB_PATHS is used by our patched package_utils.py to find DLLs
$Env:OPENVINO_LIB_PATHS = "$ReleaseBin;$TbbBin"

# 4. Set PYTHONPATH to point DIRECTLY to the compiled python package
# This bypasses the need for 'pip install -e .' and ensures we test the binaries we just built.
$PythonPackageDir = Join-Path $BuildDir "python_package" 

# Helper to avoid duplicate PYTHONPATH entries
function Add-ToPythonPath {
    param($PathToAdd)
    if ($Env:PYTHONPATH -split ';' -notcontains $PathToAdd) {
        $Env:PYTHONPATH = "$PathToAdd;" + $Env:PYTHONPATH
    }
}

# Depending on CMake flags, it might be in 'python' or 'python_package'. 
# We add the root of the python package so 'import openvino' works.
if (Test-Path (Join-Path $ReleaseBin "python_package")) {
    Add-ToPythonPath (Join-Path $ReleaseBin "python_package")
} elseif (Test-Path (Join-Path $RepoRoot "src\bindings\python\src")) {
    # Fallback to source + binary mix if cmake install wasn't run
    Write-Warning "Using split source/binary pathing."
    Add-ToPythonPath "$ReleaseBin\python"
    # Add-ToPythonPath "$RepoRoot\src\bindings\python\src"
    # Add OVC (OpenVINO Converter) tool
    Add-ToPythonPath "$RepoRoot\tools\ovc"
    # Add Benchmark Tool
    Add-ToPythonPath "$RepoRoot\tools\benchmark_tool"
}

# 5. Diagnostic Output
Write-Host ">>> Environment Configured." -ForegroundColor Green
Write-Host "    PATH Prepended: $ReleaseBin"
Write-Host "    PYTHONPATH:     $Env:PYTHONPATH"

# 6. Verify Import (Sanity Check)
python -c "import os; os.add_dll_directory(r'$ReleaseBin') if hasattr(os, 'add_dll_directory') else None; os.add_dll_directory(r'$TbbBin') if hasattr(os, 'add_dll_directory') else None; import openvino; print(f'SUCCESS: OpenVINO loaded from {openvino.__file__}')"
