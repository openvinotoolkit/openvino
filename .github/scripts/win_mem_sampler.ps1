# Memory sampler for Windows CI builds (DRAFT, experiment only).
#
# Every $IntervalSec seconds it records
#   * system commit charge / commit limit / commit peak (psapi!GetPerformanceInfo) and available physical memory;
#     "compiler is out of heap space" (C1060) is a commit exhaustion, so commit is the number that matters;
#   * per-process private bytes (= commit charge) and working set of the build tools;
# and when a build tool process exits it records its peak and the source file / output it was working on
# (taken once from the command line when the process is first seen).
#
# Output (in -OutDir):
#   samples.csv    one row per tick, flushed immediately (survives a killed job)
#   processes.csv  one row per finished (or still running at stop) tool process with its peak memory
# Stop: create the file <OutDir>\stop; the sampler flushes the still-running processes and exits.
#
# Compatible with Windows PowerShell 5.1 and PowerShell 7.
param(
    [Parameter(Mandatory = $true)][string]$OutDir,
    [double]$IntervalSec = 1.0,
    [string[]]$ProcessNames = @('cl', 'link', 'lib', 'mspdbsrv', 'cmake', 'ninja', 'MSBuild', 'rc', 'mt', 'python')
)

$ErrorActionPreference = 'Continue'
[System.Threading.Thread]::CurrentThread.CurrentCulture = [System.Globalization.CultureInfo]::InvariantCulture
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$stopFile = Join-Path $OutDir 'stop'
$samplesPath = Join-Path $OutDir 'samples.csv'
$procsPath = Join-Path $OutDir 'processes.csv'

$perfInfoAvailable = $false
try {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class OvPerfInfo {
    [StructLayout(LayoutKind.Sequential)]
    public struct PERFORMANCE_INFORMATION {
        public uint cb;
        public UIntPtr CommitTotal, CommitLimit, CommitPeak, PhysicalTotal, PhysicalAvailable,
                       SystemCache, KernelTotal, KernelPaged, KernelNonpaged, PageSize;
        public uint HandleCount, ProcessCount, ThreadCount;
    }
    [DllImport("psapi.dll", SetLastError = true)]
    static extern bool GetPerformanceInfo(out PERFORMANCE_INFORMATION pi, uint size);

    // returns pages-based values converted to bytes: commit, limit, peak, physTotal, physAvailable
    public static double[] Get() {
        PERFORMANCE_INFORMATION pi;
        if (!GetPerformanceInfo(out pi, (uint)Marshal.SizeOf(typeof(PERFORMANCE_INFORMATION))))
            return null;
        double page = (double)pi.PageSize.ToUInt64();
        return new double[] {
            pi.CommitTotal.ToUInt64() * page, pi.CommitLimit.ToUInt64() * page, pi.CommitPeak.ToUInt64() * page,
            pi.PhysicalTotal.ToUInt64() * page, pi.PhysicalAvailable.ToUInt64() * page };
    }
}
'@
    $perfInfoAvailable = $true
} catch {
    Write-Warning "GetPerformanceInfo is not available, system columns will be 0: $_"
}

$GB = 1GB
$MB = 1MB

function Get-SysMem {
    if ($perfInfoAvailable) {
        try {
            $v = [OvPerfInfo]::Get()
            if ($v) { return $v }
        } catch {
            Write-Warning "GetPerformanceInfo failed, system columns will be 0: $_"
            $script:perfInfoAvailable = $false
        }
    }
    return @(0, 0, 0, 0, 0)
}

# pid -> @{ Name; Source; First; Last; PeakPrivate; PeakWs }
$tracked = @{}

function Get-Source([int]$procId, [string]$name) {
    try {
        $cl = (Get-CimInstance Win32_Process -Filter "ProcessId=$procId" -Property CommandLine -ErrorAction Stop).CommandLine
    } catch { return '' }
    if (-not $cl) { return '' }
    if ($name -eq 'cl') {
        if ($cl -match '\s-c\s+("?[^"\s]+"?)') { return $Matches[1] }
        if ($cl -match '/Fo("?[^"\s]+"?)') { return $Matches[1] }
    } elseif ($name -eq 'link' -or $name -eq 'lib') {
        if ($cl -match '/out:("?[^"\s]+"?)') { return $Matches[1] }
    }
    if ($cl.Length -gt 200) { return $cl.Substring(0, 200) }
    return $cl
}

function Write-ProcessRow($procId, $info, [double]$now) {
    $line = '{0},{1},{2:F0},{3:F0},{4:F1},{5:F1},"{6}"' -f $procId, $info.Name, $info.First, ($info.Last - $info.First),
        ($info.PeakPrivate / $MB), ($info.PeakWs / $MB), ($info.Source -replace '"', "'")
    Add-Content -Path $procsPath -Value $line
}

Set-Content -Path $samplesPath -Value 'time_utc,t_sec,commit_gb,commit_limit_gb,commit_peak_gb,phys_avail_gb,phys_total_gb,cl_n,cl_private_gb,link_n,link_private_gb,mspdbsrv_n,mspdbsrv_private_gb,tools_private_gb,top_processes'
Set-Content -Path $procsPath -Value 'pid,name,first_seen_t_sec,seen_for_sec,peak_private_mb,peak_working_set_mb,source'

$sw = [System.Diagnostics.Stopwatch]::StartNew()
$writer = New-Object System.IO.StreamWriter($samplesPath, $true)
$writer.AutoFlush = $true
try {
    while (-not (Test-Path $stopFile)) {
        $t0 = $sw.Elapsed.TotalSeconds
        $sys = Get-SysMem
        $procs = @(Get-Process -Name $ProcessNames -ErrorAction SilentlyContinue)
        $seen = @{}
        foreach ($p in $procs) {
            $procId = $p.Id
            $seen[$procId] = $true
            try { $priv = [double]$p.PrivateMemorySize64; $ws = [double]$p.WorkingSet64 } catch { continue }
            if (-not $tracked.ContainsKey($procId)) {
                $tracked[$procId] = @{
                    Name = $p.ProcessName; Source = (Get-Source $procId $p.ProcessName)
                    First = $t0; Last = $t0; PeakPrivate = 0.0; PeakWs = 0.0
                }
            }
            $info = $tracked[$procId]
            $info.Last = $t0
            if ($priv -gt $info.PeakPrivate) { $info.PeakPrivate = $priv }
            if ($ws -gt $info.PeakWs) { $info.PeakWs = $ws }
            $info.Cur = $priv
        }
        # finished processes -> processes.csv
        foreach ($procId in @($tracked.Keys)) {
            if (-not $seen.ContainsKey($procId)) {
                Write-ProcessRow $procId $tracked[$procId] $t0
                $tracked.Remove($procId)
            }
        }
        # aggregates
        $agg = @{}
        foreach ($n in 'cl', 'link', 'mspdbsrv') { $agg[$n] = @{ N = 0; Sum = 0.0 } }
        $toolsSum = 0.0
        $rows = @()
        foreach ($procId in $tracked.Keys) {
            $i = $tracked[$procId]
            if (-not $i.ContainsKey('Cur')) { continue }
            $toolsSum += $i.Cur
            if ($agg.ContainsKey($i.Name)) { $agg[$i.Name].N++; $agg[$i.Name].Sum += $i.Cur }
            $rows += [pscustomobject]@{ Name = $i.Name; Id = $procId; Cur = $i.Cur }
        }
        $top = ($rows | Sort-Object Cur -Descending | Select-Object -First 8 |
            ForEach-Object { '{0}:{1}:{2:F0}MB' -f $_.Name, $_.Id, ($_.Cur / $MB) }) -join ' '
        $line = '{0},{1:F1},{2:F2},{3:F2},{4:F2},{5:F2},{6:F2},{7},{8:F2},{9},{10:F2},{11},{12:F2},{13:F2},"{14}"' -f
            [DateTime]::UtcNow.ToString('o'), $t0, ($sys[0] / $GB), ($sys[1] / $GB), ($sys[2] / $GB), ($sys[4] / $GB), ($sys[3] / $GB),
            $agg['cl'].N, ($agg['cl'].Sum / $GB), $agg['link'].N, ($agg['link'].Sum / $GB),
            $agg['mspdbsrv'].N, ($agg['mspdbsrv'].Sum / $GB), ($toolsSum / $GB), $top
        $writer.WriteLine($line)

        $sleepMs = [int](1000 * $IntervalSec - 1000 * ($sw.Elapsed.TotalSeconds - $t0))
        if ($sleepMs -gt 0) { Start-Sleep -Milliseconds $sleepMs }
    }
} finally {
    foreach ($procId in @($tracked.Keys)) {
        Write-ProcessRow $procId $tracked[$procId] $sw.Elapsed.TotalSeconds
    }
    $writer.Dispose()
}
