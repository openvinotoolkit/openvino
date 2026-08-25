# Bootstrap of the standalone test toolchain (no Vulkan SDK install needed).
#
# Installs into %LOCALAPPDATA%\vktools:
#   glslang\bin\glslang.exe      — GLSL -> SPIR-V (KhronosGroup/glslang release)
#   vulkan\include\vulkan\*.h    — Vulkan-Headers
#   vulkan\lib\vulkan-1.lib      — import library generated from the system dll
#
# Run:  powershell -ExecutionPolicy Bypass -File tools\setup_vktools.ps1

$ErrorActionPreference = "Stop"
$root = Join-Path $env:LOCALAPPDATA "vktools"
New-Item -ItemType Directory -Force -Path $root | Out-Null

# ---- 1. glslang -------------------------------------------------------------
$glslangVersion = "16.5.0"
$glslangDir = Join-Path $root "glslang"
if (-not (Test-Path (Join-Path $glslangDir "bin\glslang.exe"))) {
    $zip = Join-Path $root "glslang.zip"
    $url = "https://github.com/KhronosGroup/glslang/releases/download/$glslangVersion/glslang-$glslangVersion-windows-x86_64-release.zip"
    Write-Host "downloading $url"
    Invoke-WebRequest -Uri $url -OutFile $zip -UserAgent "opencode"
    Expand-Archive -LiteralPath $zip -DestinationPath $glslangDir -Force
    Remove-Item $zip
}

# ---- 2. Vulkan headers ------------------------------------------------------
$hdrVersion = "1.3.290"
$incDst = Join-Path $root "vulkan\include"
if (-not (Test-Path (Join-Path $incDst "vulkan\vulkan.h"))) {
    $zip = Join-Path $root "vulkan-headers.zip"
    $url = "https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/v$hdrVersion.zip"
    Write-Host "downloading $url"
    Invoke-WebRequest -Uri $url -OutFile $zip -UserAgent "opencode"
    $tmp = Join-Path $root "vh"
    Expand-Archive -LiteralPath $zip -DestinationPath $tmp -Force
    Copy-Item -Recurse -Force -Path (Join-Path $tmp "Vulkan-Headers-$hdrVersion\include") -Destination $incDst
    Remove-Item $zip; Remove-Item -Recurse -Force $tmp
}

# ---- 3. vulkan-1.lib from the system dll ------------------------------------
$lib = Join-Path $root "vulkan\lib\vulkan-1.lib"
if (-not (Test-Path $lib)) {
    New-Item -ItemType Directory -Force -Path (Split-Path $lib) | Out-Null

    # locate dumpbin/lib through any VS 2022 installation
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found (VS 2022 required)" }
    $vsRoot = & $vswhere -latest -products * -property installationPath
    $tools = Get-ChildItem -Recurse -Filter dumpbin.exe -LiteralPath (Join-Path $vsRoot "VC\Tools\MSVC") |
        Where-Object { $_.FullName -match "Hostx64\\x64" } | Select-Object -First 1
    if (-not $tools) { throw "dumpbin.exe not found under $vsRoot" }
    $bindir = Split-Path $tools.FullName

    $exports = & (Join-Path $bindir "dumpbin.exe") /EXPORTS "$env:SystemRoot\System32\vulkan-1.dll" | Out-String
    $names = @()
    foreach ($line in ($exports -split "`r?`n")) {
        if ($line -match "^\s+\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(\S+)\s*$") { $names += $Matches[1] }
    }
    if ($names.Count -eq 0) { throw "no exports parsed from vulkan-1.dll" }

    $def = Join-Path $root "vulkan.def"
    Set-Content -LiteralPath $def -Value (("LIBRARY vulkan-1.dll`r`nEXPORTS`r`n") +
        (($names | ForEach-Object { "    $_" }) -join "`r`n")) -Encoding ASCII

    $vcvars = Join-Path $vsRoot "VC\Auxiliary\Build\vcvars64.bat"
    cmd /c "call `"$vcvars`" >nul && lib /def:`"$def`" /machine:x64 /out:`"$lib`" /nologo" | Out-Null
    Remove-Item $def
}
if (-not (Test-Path $lib)) { throw "vulkan-1.lib was not generated" }

Write-Host "vktools ready at $root"
