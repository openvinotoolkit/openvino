# Link probe for Windows CI builds (DRAFT, experiment only).
#
# Why: with identical CMake options and the same list of static libraries, link.exe of a sample (hello_classification)
# peaks at ~7.0 GB when the build runs through "Ninja Multi-Config" and at ~3.1 GB through "Visual Studio 17 2022"
# (see the memory sampler data). The compile flags and link inputs of the Ninja build live in response files that
# `ninja -v` does not print, so this script records what the logs miss. It runs right after the build, on the same
# build tree, so everything is directly comparable between the two generators:
#   env.txt                 environment of the job (LINK / _LINK_ / CL / _CL_ and friends change tool behavior)
#   artifact_sizes.csv      size of every lib / exe / dll / pyd in the output folder (are the inputs bigger?)
#   section_sizes_<lib>.csv per-section raw size summed over the members of the biggest static libraries
#                           (.debug$S / .debug$T = debug info in objects, .text$mn = /Gy COMDATs, .drectve, ...)
#   *.rsp                   response files of the probe target (Ninja only) and of one compile step
#   relink_<target>.log     second relink with `/time+ /verbose:lib` (pass timings, which objects get pulled from libs)
# The first relink is plain, so its link.exe shows up in the memory sampler data as a clean measurement.
# Compatible with Windows PowerShell 5.1 and PowerShell 7.
param(
    [Parameter(Mandatory = $true)][string]$BuildDir,
    [Parameter(Mandatory = $true)][string]$BinDir,
    [Parameter(Mandatory = $true)][string]$OutDir,
    [string]$Target = 'hello_classification',
    [string]$Config = 'Release',
    [int]$Parallel = 4
)

$ErrorActionPreference = 'Continue'
[System.Threading.Thread]::CurrentThread.CurrentCulture = [System.Globalization.CultureInfo]::InvariantCulture
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$summary = New-Object System.Collections.Generic.List[string]
$summary.Add('### Link probe')

# 1. environment (tokens and runner internals left out)
Get-ChildItem Env: |
    Where-Object { $_.Name -notmatch 'TOKEN|SECRET|PASSWORD|CREDENTIAL|KEY|^ACTIONS_|^GITHUB_|^RUNNER_' } |
    Sort-Object Name | ForEach-Object { '{0}={1}' -f $_.Name, $_.Value } | Set-Content -Path (Join-Path $OutDir 'env.txt')
foreach ($n in '_LINK_', 'LINK', '_CL_', 'CL', 'CXXFLAGS', 'CFLAGS', 'LDFLAGS') {
    $v = [Environment]::GetEnvironmentVariable($n)
    if ($v) { $summary.Add("- env $n = ``$v``") }
}

# 2. sizes of the build outputs
$files = @(Get-ChildItem -Path $BinDir -File -ErrorAction SilentlyContinue | Where-Object { '.lib', '.exe', '.dll', '.pyd' -contains $_.Extension })
$files | Sort-Object Length -Descending | Select-Object Name, Length |
    Export-Csv -Path (Join-Path $OutDir 'artifact_sizes.csv') -NoTypeInformation
$libs = @($files | Where-Object { $_.Extension -eq '.lib' })
$libBytes = ($libs | Measure-Object -Property Length -Sum).Sum
$summary.Add(('- static / import libs in the output folder: {0}, total {1:F2} GB' -f $libs.Count, ($libBytes / 1GB)))
$exe = Join-Path $BinDir "$Target.exe"
if (Test-Path $exe) { $summary.Add(('- {0}.exe size: {1:F1} MB' -f $Target, ((Get-Item $exe).Length / 1MB))) }

# 3. response files (Ninja only; the VS generator passes everything on the command line that is already in the log)
Get-ChildItem -Path $BuildDir -Recurse -Filter "$Target.$Config.rsp" -ErrorAction SilentlyContinue | ForEach-Object {
    Copy-Item $_.FullName (Join-Path $OutDir $_.Name)
}
Get-ChildItem -Path $BuildDir -Recurse -Filter 'model.cpp.obj.rsp' -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match 'openvino_core_obj' } | Select-Object -First 1 | ForEach-Object {
        Copy-Item $_.FullName (Join-Path $OutDir 'compile_core_model.cpp.obj.rsp')
    }

# 4. what is inside the biggest static libraries (debug info, COMDAT sections, ...)
$dumpbin = $null
if ($env:VCToolsInstallDir) { $dumpbin = Join-Path $env:VCToolsInstallDir 'bin\Hostx64\x64\dumpbin.exe' }
if ($dumpbin -and (Test-Path $dumpbin)) {
    $probeLibs = @($libs | Sort-Object Length -Descending | Select-Object -First 2)
    $ref = $libs | Where-Object { $_.Name -eq 'openvino_reference.lib' } | Select-Object -First 1
    if ($ref -and -not ($probeLibs | Where-Object { $_.Name -eq $ref.Name })) { $probeLibs += $ref }
    foreach ($lib in $probeLibs) {
        $sizes = @{}; $counts = @{}; $name = $null
        & $dumpbin /nologo /headers $lib.FullName 2>&1 | ForEach-Object {
            if ($_ -match '^\s+(\S+) name$') { $name = $Matches[1] }
            elseif ($_ -match '^\s+([0-9A-Fa-f]+) size of raw data$' -and $name) {
                $sizes[$name] = [double]$sizes[$name] + [Convert]::ToInt64($Matches[1], 16)
                $counts[$name] = [int]$counts[$name] + 1
                $name = $null
            }
        }
        $csv = Join-Path $OutDir ('section_sizes_{0}.csv' -f $lib.BaseName)
        $sizes.GetEnumerator() | Sort-Object Value -Descending |
            ForEach-Object { [pscustomobject]@{ section = $_.Key; members = $counts[$_.Key]; raw_mb = [math]::Round($_.Value / 1MB, 1) } } |
            Export-Csv -Path $csv -NoTypeInformation
        $top = Import-Csv $csv | Select-Object -First 6 | ForEach-Object { '{0}: {1} MB' -f $_.section, $_.raw_mb }
        $dbg = Import-Csv $csv | Where-Object { $_.section -like '.debug*' } | ForEach-Object { '{0}: {1} MB' -f $_.section, $_.raw_mb }
        $summary.Add(('- {0} ({1:F0} MB) biggest sections: {2}; debug sections: {3}' -f $lib.Name, ($lib.Length / 1MB), ($top -join ', '), $(if ($dbg) { $dbg -join ', ' } else { 'none' })))
    }
} else {
    $summary.Add('- dumpbin was not found (VCToolsInstallDir is not set), section sizes skipped')
}

# 5. relink the probe target with the native generator: first plain, then with linker timing and library tracing
$gen = ''
$cache = Join-Path $BuildDir 'CMakeCache.txt'
if (Test-Path $cache) {
    $m = Select-String -Path $cache -Pattern '^CMAKE_GENERATOR:INTERNAL=(.*)$' | Select-Object -First 1
    if ($m) { $gen = $m.Matches[0].Groups[1].Value }
}
$native = @('/verbosity:minimal')
if ($gen -like 'Ninja*') { $native = @('-v') }
$summary.Add("- generator: $gen")

foreach ($variant in 'plain', 'verbose') {
    if (-not (Test-Path $exe)) { $summary.Add("- relink ($variant) skipped: $exe does not exist"); continue }
    Remove-Item $exe -Force
    $log = Join-Path $OutDir ("relink_{0}_{1}.log" -f $Target, $variant)
    if ($variant -eq 'verbose') { $env:_LINK_ = '/time+ /verbose:lib' }
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & cmake --build $BuildDir --config $Config --target $Target --parallel $Parallel -- @native *>&1 | Out-File -FilePath $log -Encoding utf8
    $sw.Stop()
    Remove-Item Env:\_LINK_ -ErrorAction SilentlyContinue
    $summary.Add(('- relink ({0}): cmake --build took {1:F0} s, exit code {2}, exe recreated: {3}' -f $variant, $sw.Elapsed.TotalSeconds, $LASTEXITCODE, (Test-Path $exe)))
    if ($variant -eq 'verbose') {
        $loaded = 0
        foreach ($l in [System.IO.File]::ReadLines($log)) { if ($l -match '^\s+Loaded ') { $loaded++ } }
        $summary.Add("- relink (verbose): objects loaded from libraries: $loaded")
        Select-String -Path $log -Pattern 'Pass|[Ww]all clock|Total time|ICF|Linker' | Select-Object -First 30 | ForEach-Object {
            $summary.Add("    $($_.Line.Trim())")
        }
    }
}

$summary | Out-File -Append -Encoding utf8 $env:GITHUB_STEP_SUMMARY
$summary | Write-Output
