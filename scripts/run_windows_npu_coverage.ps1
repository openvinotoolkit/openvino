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
      <Include><Source>.*[\\/]src[\\/]plugins[\\/]intel_npu[\\/].*</Source></Include>
      <Exclude><Source>.*[\\/]tests[\\/].*</Source><Source>.*[\\/]thirdparty[\\/].*</Source></Exclude>
    </Sources>
  </CodeCoverage>
</Configuration>
"@
    switch ($Mode) {
        'Plain' {
            & $exe '--gtest_filter=*' "--gtest_output=xml:$testXml" 2>&1 |
                Tee-Object "$ResultsDir/run.log"
        }
        'Console' {
            $settings = Join-Path $ResultsDir 'coverage.config'
            $configuration | Set-Content $settings
            & $ToolPath collect --settings $settings --output "$ResultsDir/coverage.cobertura.xml" `
                --output-format cobertura --log-file "$ResultsDir/collector.log" --log-level Verbose `
                $exe '--gtest_filter=*' "--gtest_output=xml:$testXml" 2>&1 |
                Tee-Object "$ResultsDir/run.log"
        }
        'VSTest' {
            $settings = Join-Path $ResultsDir 'coverage.runsettings'
            @"
<RunSettings>
  <RunConfiguration><TargetPlatform>x64</TargetPlatform><TreatNoTestsAsError>true</TreatNoTestsAsError></RunConfiguration>
  <DataCollectionRunSettings><DataCollectors>
    <DataCollector friendlyName="Code Coverage">$configuration</DataCollector>
  </DataCollectors></DataCollectionRunSettings>
</RunSettings>
"@ | Set-Content $settings
            & $ToolPath $exe "/TestAdapterPath:$AdapterPath" "/Settings:$settings" `
                '/EnableCodeCoverage' "/ResultsDirectory:$ResultsDir" '/Logger:trx;LogFileName=tests.trx' `
                "/Diag:$ResultsDir/vstest.log" 2>&1 | Tee-Object "$ResultsDir/run.log"
        }
    }
    $result = $LASTEXITCODE
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
        $covered = 0
        foreach ($report in $reports) {
            [xml]$coverage = Get-Content $report.FullName -Raw
            $covered += @($coverage.SelectNodes('/coverage/packages/package/classes/class/lines/line[@hits > 0]')).Count
        }
        if ($covered -eq 0) { throw 'No covered NPU source lines in Cobertura output; inspect collector logs.' }
        Write-Host "Covered line entries (not deduplicated across modules): $covered"
    }
    if ($result -ne 0) { throw "Test/collector exit code: $result. Reports have been preserved." }
    if ($failedTests -gt 0) { throw "$failedTests failed tests in XML results. Reports have been preserved." }
} finally {
    Pop-Location
    $env:PATH = $savedPath
    foreach ($name in $savedTestEnv.Keys) { Set-Item "Env:$name" $savedTestEnv[$name] }
}
