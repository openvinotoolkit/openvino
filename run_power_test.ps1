$counter = "\Energy Meter(RAPL_Package0_PKG)\Power"
$csvFile = "$PWD\power_data.csv"

# Remove old file if exists
if (Test-Path $csvFile) { Remove-Item $csvFile }

Write-Host "Starting Power Measurement..."
# Start typeperf in background, collecting every 1 second
$job = Start-Job -ScriptBlock { 
    param($c, $f) 
    typeperf $c -si 1 -sc 60 -f CSV -o $f -y 
} -ArgumentList $counter, $csvFile

# Wait for typeperf to initialize
Start-Sleep -Seconds 3

# Debug: Check if job is running
$j = Get-Job $job.Id
Write-Host "Job State: $($j.State)"
if ($j.State -eq 'Failed') {
    Write-Host "Job Failed. Reason:"
    Receive-Job $job
}

Write-Host "Starting Benchmark..."
# Run the benchmark
python run_benchmark.py

Write-Host "Benchmark Complete. Stopping Power Measurement..."
# Stop the job
Stop-Job $job
Remove-Job $job

# Analyze Data
if (Test-Path $csvFile) {
    $data = Import-Csv $csvFile
    # The column name might be complex, so we get the second property (first is timestamp)
    $colName = $data[0].PSObject.Properties.Name[1]
    
    $values = $data | ForEach-Object { 
        $val = $_.$colName
        if ($val -ne $null -and $val -ne "") {
            [double]$val
        }
    }
    
    if ($values.Count -gt 0) {
        $avgPowerMW = ($values | Measure-Object -Average).Average
        $avgPowerW = $avgPowerMW / 1000.0
        Write-Host "Average Power Consumption: $avgPowerW Watts"
        
        # Also show min/max
        $stats = $values | Measure-Object -Minimum -Maximum
        $minW = $stats.Minimum / 1000.0
        $maxW = $stats.Maximum / 1000.0
        Write-Host "Min Power: $minW W"
        Write-Host "Max Power: $maxW W"
    } else {
        Write-Host "No power data collected."
    }
} else {
    Write-Host "Error: Power data file not found."
}
