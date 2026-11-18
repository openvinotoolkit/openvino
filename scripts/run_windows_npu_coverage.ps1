# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Local experiment only; no NPU device preflight, CI lane split, or Codecov upload.
param(
    [ValidateSet('Plain', 'VSTest', 'Console')][string]$Mode = 'Plain',
    [string]$PackageDir = "$PSScriptRoot/../build-windows-npu-coverage/package",
    [string]$ToolPath,
    [string]$AdapterPath,
    [ValidateSet('Dynamic', 'Static')][string]$Instrumentation = 'Dynamic',
    [string]$ResultsDir = "$PSScriptRoot/../build-windows-npu-coverage/results/$Mode-$(Get-Date -Format 'yyyyMMdd-HHmmss-fff')"
)

$ErrorActionPreference = 'Stop'
function Invoke-TestTool {
    param([string]$FilePath, [string[]]$Arguments, [string]$LogPath)
    Get-Command $FilePath -ErrorAction Stop | Out-Null
    # Windows PowerShell 5.1 turns native stderr into ErrorRecords. Log them and
    # check the exit code/results afterwards instead of aborting the child early.
    $ErrorActionPreference = 'Continue'
    $PSNativeCommandUseErrorActionPreference = $false
    & $FilePath @Arguments 2>&1 |
        ForEach-Object { $_.ToString() } -ErrorAction Stop |
        Tee-Object -FilePath $LogPath -ErrorAction Stop | Out-Host
    return $LASTEXITCODE
}

function Get-NpuCoverageSummary {
    param([IO.FileInfo[]]$Reports, [string]$SourceRoot)
    $hashes = @{}
    $lines = @{}
    $files = @{}
    $modulesWithHits = @{}
    $expectedModules = @('ov_npu_unit_tests', 'openvino_intel_npu_plugin')
    foreach ($report in $Reports) {
        $hash = (Get-FileHash $report.FullName -Algorithm SHA256).Hash
        if ($hashes.ContainsKey($hash)) { continue }
        $hashes[$hash] = $true
        [xml]$coverage = Get-Content $report.FullName -Raw
        foreach ($module in $coverage.coverage.packages.package) {
            $name = [IO.Path]::GetFileNameWithoutExtension($module.name)
            if ($name -notin $expectedModules) { continue }
            foreach ($class in $module.classes.class) {
                $path = $class.filename.Replace('\', '/')
                if ($path -notmatch '^(?:.*?/)?(src/plugins/intel_npu/.*)$') { continue }
                $source = $Matches[1]
                if ($source -match '/(tests|thirdparty|\.\.)/' -or
                    !(Test-Path -LiteralPath (Join-Path $SourceRoot $source) -PathType Leaf)) { continue }
                foreach ($line in $class.lines.line) {
                    $key = "${source}:$($line.number)"
                    $hit = [long]$line.hits -gt 0
                    $lines[$key] = $lines[$key] -or $hit
                    $files[$source] = $true
                    if ($hit) { $modulesWithHits[$name] = $true }
                }
            }
        }
    }
    foreach ($name in $expectedModules) {
        if (!$modulesWithHits.ContainsKey($name)) {
            throw "No covered production NPU lines from $name; inspect source filters, PDBs and collector logs."
        }
    }
    [pscustomobject]@{
        Covered = @($lines.Values | Where-Object { $_ }).Count
        Total = $lines.Count
        Files = $files.Count
    }
}

$PackageDir = (Resolve-Path $PackageDir).Path
$modules = foreach ($name in @('ov_npu_unit_tests.exe', 'openvino_intel_npu_plugin.dll')) {
    $found = @(Get-ChildItem $PackageDir -Recurse -File -Filter $name)
    if ($found.Count -ne 1) { throw "Expected exactly one $name in $PackageDir" }
    if (!(Test-Path ([IO.Path]::ChangeExtension($found[0].FullName, '.pdb')))) {
        throw "Missing PDB for $($found[0].FullName)"
    }
    $found[0]
}
if ($Mode -ne 'Plain') {
    if (!$ToolPath) { throw "-ToolPath is required for $Mode" }
    $ToolPath = (Resolve-Path $ToolPath).Path
}
if ($Mode -eq 'VSTest') {
    if (!$AdapterPath) { throw '-AdapterPath must point to the Google Test adapter DLL directory' }
    $AdapterPath = (Resolve-Path $AdapterPath).Path
}
$ResultsDir = [IO.Path]::GetFullPath($ResultsDir)
if (Test-Path $ResultsDir) { throw "Use a fresh results directory: $ResultsDir" }
New-Item -ItemType Directory -Path $ResultsDir | Out-Null

$savedPath = $env:PATH
$savedTestEnv = @{}
Get-ChildItem Env: | Where-Object { $_.Name -like 'GTEST_*' -or $_.Name -eq 'OV_DQ_TEST_DEVICE' } |
    ForEach-Object {
        $savedTestEnv[$_.Name] = $_.Value
        Remove-Item "Env:$($_.Name)"
    }
# Avoid setupvars.ps1's Python checks/early Exit: only native DLL lookup is needed here.
$dllDirs = Get-ChildItem "$PackageDir/runtime" -Recurse -File -Filter '*.dll' |
    Select-Object -ExpandProperty DirectoryName -Unique
$env:PATH = (@($modules[0].DirectoryName) + @($dllDirs) + @($savedPath)) -join [IO.Path]::PathSeparator
Push-Location $modules[0].DirectoryName
try {
    $exe = $modules[0].FullName
    $testXml = Join-Path $ResultsDir 'tests.xml'
    $static = ($Instrumentation -eq 'Static').ToString()
    $dynamic = ($Instrumentation -eq 'Dynamic').ToString()
    $directories = ($modules | Select-Object -ExpandProperty DirectoryName -Unique |
        ForEach-Object { '<Directory Recursive="False">' + [Security.SecurityElement]::Escape($_) + '</Directory>' }) -join ''
    $configuration = @"
<Configuration>
  <Format>cobertura</Format>
  <IncludeTestAssembly>True</IncludeTestAssembly>
  <ExcludeAssembliesWithoutSources>None</ExcludeAssembliesWithoutSources>
  <CodeCoverage>
    <EnableStaticNativeInstrumentation>$static</EnableStaticNativeInstrumentation>
    <EnableDynamicNativeInstrumentation>$dynamic</EnableDynamicNativeInstrumentation>
    <EnableStaticNativeInstrumentationRestore>True</EnableStaticNativeInstrumentationRestore>
    <EnableStaticManagedInstrumentation>False</EnableStaticManagedInstrumentation>
    <EnableDynamicManagedInstrumentation>False</EnableDynamicManagedInstrumentation>
    <ModulePaths>
      <Include><ModulePath>.*[\\/]ov_npu_unit_tests\.exe$</ModulePath><ModulePath>.*[\\/]openvino_intel_npu_plugin\.dll$</ModulePath></Include>
      <IncludeDirectories>$directories</IncludeDirectories>
    </ModulePaths>
    <Sources>
      <Include><Source>(^|.*[\\/])src[\\/]plugins[\\/]intel_npu[\\/].*</Source></Include>
      <Exclude><Source>.*[\\/]tests[\\/].*</Source><Source>.*[\\/]thirdparty[\\/].*</Source></Exclude>
    </Sources>
  </CodeCoverage>
</Configuration>
"@
    switch ($Mode) {
        'Plain' {
            $result = Invoke-TestTool $exe @('--gtest_filter=*', "--gtest_output=xml:$testXml") "$ResultsDir/run.log"
        }
        'Console' {
            $settings = Join-Path $ResultsDir 'coverage.config'
            $configuration | Set-Content $settings
            $result = Invoke-TestTool $ToolPath @(
                'collect', '--settings', $settings, '--output', "$ResultsDir/coverage.cobertura.xml",
                '--output-format', 'cobertura', '--log-file', "$ResultsDir/collector.log", '--log-level', 'Verbose',
                $exe, '--gtest_filter=*', "--gtest_output=xml:$testXml"
            ) "$ResultsDir/run.log"
        }
        'VSTest' {
            $settings = Join-Path $ResultsDir 'coverage.runsettings'
            @"
<RunSettings>
  <RunConfiguration><TargetPlatform>x64</TargetPlatform><TreatNoTestsAsError>true</TreatNoTestsAsError></RunConfiguration>
  <GoogleTestAdapterSettings>
    <SolutionSettings><Settings>
      <UseNewTestExecutionFramework>false</UseNewTestExecutionFramework>
      <ParseSymbolInformation>false</ParseSymbolInformation>
    </Settings></SolutionSettings>
  </GoogleTestAdapterSettings>
  <DataCollectionRunSettings><DataCollectors>
    <DataCollector friendlyName="Code Coverage">$configuration</DataCollector>
  </DataCollectors></DataCollectionRunSettings>
</RunSettings>
"@ | Set-Content $settings
            $result = Invoke-TestTool $ToolPath @(
                $exe, "/TestAdapterPath:$AdapterPath", "/Settings:$settings",
                '/EnableCodeCoverage', "/ResultsDirectory:$ResultsDir", '/Logger:trx;LogFileName=tests.trx',
                "/Diag:$ResultsDir/vstest.log"
            ) "$ResultsDir/run.log"
        }
    }
    "Exit code: $result" | Tee-Object "$ResultsDir/exit-code.txt"
    Write-Host "Results and diagnostics: $ResultsDir"
    if ($Mode -eq 'VSTest') {
        [xml]$tests = Get-Content "$ResultsDir/tests.trx" -Raw
        if ([int]$tests.TestRun.ResultSummary.Counters.executed -le 0) {
            throw 'VSTest did not execute any tests; check adapter discovery and vstest.log.'
        }
        $failedTests = [int]$tests.TestRun.ResultSummary.Counters.failed
    } else {
        [xml]$tests = Get-Content $testXml -Raw
        if ([int]$tests.testsuites.tests -le 0) { throw 'No NPU tests executed.' }
        $failedTests = [int]$tests.testsuites.failures + [int]$tests.testsuites.errors
    }
    if ($Mode -ne 'Plain') {
        $reports = @(Get-ChildItem $ResultsDir -Recurse -File -Filter '*cobertura.xml')
        $summary = Get-NpuCoverageSummary $reports (Resolve-Path "$PSScriptRoot/..").Path
        Write-Host ("Unique production NPU lines covered: {0}/{1} in {2} files (EXE + DLL)" -f
            $summary.Covered, $summary.Total, $summary.Files)
    }
    if ($result -ne 0) { throw "Test/collector exit code: $result. Reports have been preserved." }
    if ($failedTests -gt 0) { throw "$failedTests failed tests in XML results. Reports have been preserved." }
} finally {
    Pop-Location
    $env:PATH = $savedPath
    foreach ($name in $savedTestEnv.Keys) { Set-Item "Env:$name" $savedTestEnv[$name] }
}
