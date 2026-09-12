# Copy every .ninja_log under one or more CMake build trees into a named snapshot directory.
# Usage: collect_ninja_logs_snapshot.ps1 -ArchiveRoot <path> -SnapshotName <name> -BuildDirs <dir1>,<dir2>,...
param(
  [Parameter(Mandatory = $true)][string]$ArchiveRoot,
  [Parameter(Mandatory = $true)][string]$SnapshotName,
  [Parameter(Mandatory = $true)][string[]]$BuildDirs
)

$ErrorActionPreference = 'Stop'

function Collect-Tree {
  param(
    [string]$Root,
    [string]$Prefix
  )
  if (-not (Test-Path -LiteralPath $Root)) {
    Write-Host "Skip missing build directory: $Root"
    return
  }
  $destRoot = Join-Path $ArchiveRoot $SnapshotName
  if ($Prefix) {
    $destRoot = Join-Path $destRoot $Prefix
  }
  $logs = Get-ChildItem -LiteralPath $Root -Recurse -Force -Filter '.ninja_log' -ErrorAction SilentlyContinue
  if (-not $logs) {
    Write-Host "No .ninja_log files under $Root"
    return
  }
  Write-Host "Snapshot ${SnapshotName}: collecting $($logs.Count) .ninja_log file(s) from $Root"
  foreach ($log in $logs) {
    $rel = $log.FullName.Substring($Root.Length).TrimStart('\', '/')
    $dest = Join-Path $destRoot $rel
    $destDir = Split-Path -Parent $dest
    if (-not (Test-Path -LiteralPath $destDir)) {
      New-Item -ItemType Directory -Path $destDir -Force | Out-Null
    }
    Copy-Item -LiteralPath $log.FullName -Destination $dest -Force
  }
}

$snapshotPath = Join-Path $ArchiveRoot $SnapshotName
New-Item -ItemType Directory -Path $snapshotPath -Force | Out-Null

foreach ($dir in $BuildDirs) {
  $trimmed = $dir.Trim()
  if (-not $trimmed) { continue }
  Collect-Tree -Root $trimmed -Prefix (Split-Path -Leaf $trimmed)
}
