#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Deploy-ToColab.ps1 - Automated Cyberspore Colab Deployment

.DESCRIPTION
    Orchestrates model upload to Google Drive and result downloads
    Uses rclone for reliable file sync.

.PARAMETER Action
    Operation: UploadModel, DownloadResults, Status

.EXAMPLE
    .\deploy_to_colab.ps1 -Action UploadModel
    .\deploy_to_colab.ps1 -Action DownloadResults
    .\deploy_to_colab.ps1 -Action Status
#>

param(
    [ValidateSet('UploadModel', 'DownloadResults', 'Status')]
    [string]$Action = 'Status',
    
    [string]$RcloneRemote = 'gdrive',
    [string]$DriveFolderPath = 'Cyberspore',
    [string]$LocalModelPath = 'gemma_ir_tssn'
)

# Configuration
$Config = @{
    RcloneRemote = $RcloneRemote
    DriveFolderPath = $DriveFolderPath
    LocalModelPath = $LocalModelPath
    DriveModelPath = "$DriveFolderPath/gemma_ir_tssn"
    DriveResultsPath = "$DriveFolderPath/results"
    LocalResultsPath = './colab_results'
}

function Write-Status {
    param([string]$Message, [ValidateSet('Info', 'Success', 'Warning', 'Error')][string]$Type = 'Info')
    
    $colors = @{
        Info = 'Cyan'
        Success = 'Green'
        Warning = 'Yellow'
        Error = 'Red'
    }
    
    $time = [datetime]::Now.ToString('HH:mm:ss')
    Write-Host "[$time] $Message" -ForegroundColor $colors[$Type]
}

function Test-RcloneInstalled {
    try {
        $null = rclone --version 2>$null
        return $true
    } catch {
        return $false
    }
}

function Test-RcloneRemote {
    param([string]$Remote)
    try {
        $output = rclone ls "$Remote`:" 2>&1
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Invoke-UploadModel {
    Write-Status "=== UPLOADING MODEL TO GOOGLE DRIVE ===" -Type Info
    
    # Validate local model
    if (-not (Test-Path $Config.LocalModelPath)) {
        Write-Status "Error: Model not found at $($Config.LocalModelPath)" -Type Error
        return $false
    }
    
    $modelSize = [math]::Round((Get-ChildItem -Path $Config.LocalModelPath -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB, 1)
    Write-Status "Model size: $modelSize MB" -Type Info
    
    # Validate rclone
    if (-not (Test-RcloneInstalled)) {
        Write-Status "Error: rclone not installed" -Type Error
        return $false
    }
    
    # Check rclone remote
    if (-not (Test-RcloneRemote $Config.RcloneRemote)) {
        Write-Status "Error: rclone remote '$($Config.RcloneRemote)' not configured" -Type Error
        Write-Status "Run: rclone config" -Type Warning
        return $false
    }
    
    # Create remote folder structure
    Write-Status "Creating remote folder: $($Config.DriveFolderPath)" -Type Info
    rclone mkdir "$($Config.RcloneRemote):$($Config.DriveFolderPath)" 2>&1 | ForEach-Object { Write-Status $_ -Type Info }
    
    # Upload with progress
    Write-Status "Starting upload (this may take 5-15 minutes)..." -Type Info
    $uploadStart = [datetime]::Now
    
    rclone sync "$($Config.LocalModelPath)" "$($Config.RcloneRemote):$($Config.DriveModelPath)" `
        --progress --fast-list --transfers 4 --checkers 4 --ignore-errors 2>&1 | `
        ForEach-Object { Write-Status $_ -Type Info }
    
    if ($LASTEXITCODE -ne 0) {
        Write-Status "Warning: Upload may be incomplete" -Type Warning
    }
    
    $uploadTime = ([datetime]::Now - $uploadStart).TotalMinutes
    Write-Status "Upload completed in $([math]::Round($uploadTime, 1)) minutes" -Type Success
    
    # Verify
    Write-Status "Verifying upload..." -Type Info
    $fileCount = rclone ls "$($Config.RcloneRemote):$($Config.DriveModelPath)" --recursive 2>&1 | Measure-Object -Line
    Write-Status "Files on Drive: $($fileCount.Lines)" -Type Success
    
    return $true
}

function Invoke-DownloadResults {
    Write-Status "=== DOWNLOADING RESULTS FROM GOOGLE DRIVE ===" -Type Info
    
    # Validate rclone
    if (-not (Test-RcloneInstalled)) {
        Write-Status "Error: rclone not installed" -Type Error
        return $false
    }
    
    if (-not (Test-RcloneRemote $Config.RcloneRemote)) {
        Write-Status "Error: rclone remote not configured" -Type Error
        return $false
    }
    
    # Create local folder
    if (-not (Test-Path $Config.LocalResultsPath)) {
        New-Item -ItemType Directory -Path $Config.LocalResultsPath -Force | Out-Null
        Write-Status "Created local folder: $($Config.LocalResultsPath)" -Type Info
    }
    
    # Check if results exist on Drive
    Write-Status "Checking for results on Drive..." -Type Info
    $results = rclone ls "$($Config.RcloneRemote):$($Config.DriveResultsPath)" 2>&1
    
    if ($LASTEXITCODE -ne 0) {
        Write-Status "No results found on Drive yet" -Type Warning
        Write-Status "Please ensure Colab execution is complete" -Type Warning
        return $false
    }
    
    # Download results
    Write-Status "Starting download..." -Type Info
    $downloadStart = [datetime]::Now
    
    rclone sync "$($Config.RcloneRemote):$($Config.DriveResultsPath)" "$($Config.LocalResultsPath)" `
        --progress --fast-list --checkers 4 2>&1 | `
        ForEach-Object { Write-Status $_ -Type Info }
    
    if ($LASTEXITCODE -ne 0) {
        Write-Status "Warning: Download may be incomplete" -Type Warning
    }
    
    $downloadTime = ([datetime]::Now - $downloadStart).TotalMinutes
    Write-Status "Download completed in $([math]::Round($downloadTime, 1)) minutes" -Type Success
    
    # List downloaded files
    Write-Status "Downloaded files:" -Type Info
    Get-ChildItem -Path $Config.LocalResultsPath -Recurse | ForEach-Object {
        if (-not $_.PSIsContainer) {
            $size = [math]::Round($_.Length / 1MB, 1)
            Write-Status "  - $($_.Name) ($size MB)" -Type Info
        }
    }
    
    return $true
}

function Show-Status {
    Write-Status "=== CYBERSPORE AUTOMATION STATUS ===" -Type Info
    
    Write-Status "Configuration:" -Type Info
    Write-Host "  rclone remote: $($Config.RcloneRemote)"
    Write-Host "  Drive folder: $($Config.DriveFolderPath)"
    Write-Host "  Local model: $($Config.LocalModelPath)"
    
    Write-Status "System Status:" -Type Info
    if (Test-RcloneInstalled) {
        Write-Host "  rclone: Installed" -ForegroundColor Green
    } else {
        Write-Host "  rclone: NOT installed" -ForegroundColor Red
    }
    
    if (Test-RcloneRemote $Config.RcloneRemote) {
        Write-Host "  rclone remote: Connected" -ForegroundColor Green
    } else {
        Write-Host "  rclone remote: NOT configured" -ForegroundColor Red
    }
    
    if (Test-Path $Config.LocalModelPath) {
        $size = [math]::Round((Get-ChildItem -Path $Config.LocalModelPath -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB, 1)
        Write-Host "  Local model: Ready ($size MB)" -ForegroundColor Green
    } else {
        Write-Host "  Local model: NOT found" -ForegroundColor Red
    }
}

# Main execution
$success = $false

switch ($Action) {
    'UploadModel' {
        $success = Invoke-UploadModel
    }
    'DownloadResults' {
        $success = Invoke-DownloadResults
    }
    'Status' {
        Show-Status
        $success = $true
    }
}

if ($success) {
    Write-Status "Operation completed successfully" -Type Success
    exit 0
} else {
    Write-Status "Operation failed" -Type Error
    exit 1
}
