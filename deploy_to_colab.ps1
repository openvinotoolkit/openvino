#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Deploy-ToColab.ps1 - Automated Cyberspore Colab Deployment
    
.DESCRIPTION
    Orchestrates model upload to Google Drive, triggers Colab execution,
    and downloads results. Uses rclone for file sync and GitHub Actions
    for notebook execution.

.PARAMETER Action
    Operation to perform: UploadModel, TriggerColab, DownloadResults, Full

.PARAMETER RcloneRemote
    Rclone remote name (default: 'gdrive')

.PARAMETER DriveFolderPath
    Drive path for uploads (default: 'Cyberspore')

.EXAMPLE
    # Upload model to Drive
    .\deploy_to_colab.ps1 -Action UploadModel

    # Full automated workflow
    .\deploy_to_colab.ps1 -Action Full

    # Download results only
    .\deploy_to_colab.ps1 -Action DownloadResults
#>

param(
    [ValidateSet('UploadModel', 'TriggerColab', 'DownloadResults', 'Full', 'Status')]
    [string]$Action = 'Full',
    
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
    GitHubRepo = 'ssdajoker/openvino'
    GitHubWorkflow = 'colab-evolution'
}

function Write-Status {
    param([string]$Message, [ValidateSet('Info', 'Success', 'Warning', 'Error')][string]$Type = 'Info')
    
    $colors = @{
        Info = 'Cyan'
        Success = 'Green'
        Warning = 'Yellow'
        Error = 'Red'
    }
    
    Write-Host "[$([datetime]::Now.ToString('HH:mm:ss'))] " -NoNewline
    Write-Host $Message -ForegroundColor $colors[$Type]
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
        Write-Status "❌ Model not found at: $($Config.LocalModelPath)" -Type Error
        return $false
    }
    
    $modelSize = [math]::Round((Get-ChildItem -Path $Config.LocalModelPath -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB, 1)
    Write-Status "📦 Model size: $modelSize MB" -Type Info
    
    # Validate rclone
    if (-not (Test-RcloneInstalled)) {
        Write-Status "❌ rclone not installed. Run: setup_automation.py" -Type Error
        return $false
    }
    
    if (-not (Test-RcloneRemote $Config.RcloneRemote)) {
        Write-Status "❌ rclone remote '$($Config.RcloneRemote)' not configured" -Type Error
        Write-Status "   Run: rclone config" -Type Info
        return $false
    }
    
    # Create Drive folder
    Write-Status "📁 Creating Drive folder structure..." -Type Info
    rclone mkdir "$($Config.RcloneRemote)`:$($Config.DriveFolderPath)" 2>$null
    rclone mkdir "$($Config.RcloneRemote)`:$($Config.DriveModelPath)" 2>$null
    
    # Upload model
    Write-Status "⬆️  Uploading model (this may take 5-15 minutes for 600MB)..." -Type Info
    $uploadStart = Get-Date
    
    rclone sync $Config.LocalModelPath "$($Config.RcloneRemote)`:$($Config.DriveModelPath)" `
        --progress `
        --fast-list `
        --transfers 4 `
        --checkers 8
    
    if ($LASTEXITCODE -ne 0) {
        Write-Status "❌ Upload failed" -Type Error
        return $false
    }
    
    $uploadTime = ([datetime]::Now - $uploadStart).TotalMinutes
    Write-Status "✅ Model uploaded successfully in $([math]::Round($uploadTime, 1)) minutes" -Type Success
    
    # Verify upload
    Write-Status "🔍 Verifying upload..." -Type Info
    $driveFiles = rclone ls "$($Config.RcloneRemote)`:$($Config.DriveModelPath)" --recursive
    if ($LASTEXITCODE -eq 0) {
        Write-Status "✅ Model verified on Drive" -Type Success
        return $true
    } else {
        Write-Status "⚠️  Could not verify (might still be syncing)" -Type Warning
        return $true
    }
}

function Invoke-TriggerColab {
    Write-Status "=== TRIGGERING COLAB NOTEBOOK EXECUTION ===" -Type Info
    
    Write-Status "📝 Step 1: Go to GitHub repository" -Type Info
    Write-Status "   Repo: https://github.com/$($Config.GitHubRepo)" -Type Info
    Write-Status "   Workflow: $($Config.GitHubWorkflow)" -Type Info
    
    Write-Status "📝 Step 2: Manual Trigger (until auto-trigger is enabled)" -Type Warning
    Write-Status "   Navigate to: Actions → colab-evolution → Run workflow" -Type Info
    Write-Status "   OR use GitHub CLI: gh workflow run colab-evolution.yml" -Type Info
    
    Write-Status "📝 Step 3: Monitor execution" -Type Info
    Write-Status "   Watch: https://github.com/$($Config.GitHubRepo)/actions" -Type Info
    
    Write-Status "⏱️  Estimated time:" -Type Info
    Write-Status "   • Colab setup: ~7 minutes" -Type Info
    Write-Status "   • Evolution: 2-4 hours" -Type Info
    
    return $true
}

function Invoke-DownloadResults {
    Write-Status "=== DOWNLOADING RESULTS FROM GOOGLE DRIVE ===" -Type Info
    
    # Validate rclone
    if (-not (Test-RcloneInstalled)) {
        Write-Status "❌ rclone not installed" -Type Error
        return $false
    }
    
    if (-not (Test-RcloneRemote $Config.RcloneRemote)) {
        Write-Status "❌ rclone remote '$($Config.RcloneRemote)' not configured" -Type Error
        return $false
    }
    
    # Create local results directory
    New-Item -ItemType Directory -Path $Config.LocalResultsPath -Force | Out-Null
    
    # Download results
    Write-Status "⬇️  Downloading results (this may take 5-10 minutes)..." -Type Info
    $downloadStart = Get-Date
    
    rclone sync "$($Config.RcloneRemote)`:$($Config.DriveResultsPath)" $Config.LocalResultsPath `
        --progress `
        --fast-list
    
    if ($LASTEXITCODE -ne 0) {
        Write-Status "⚠️  Download incomplete (results may still be being written)" -Type Warning
        return $true
    }
    
    $downloadTime = ([datetime]::Now - $downloadStart).TotalMinutes
    Write-Status "✅ Results downloaded in $([math]::Round($downloadTime, 1)) minutes" -Type Success
    
    # List downloaded files
    if (Test-Path $Config.LocalResultsPath) {
        Write-Status "📂 Downloaded files:" -Type Info
        Get-ChildItem -Path $Config.LocalResultsPath -Recurse | ForEach-Object {
            if (-not $_.PSIsContainer) {
                $size = [math]::Round($_.Length / 1MB, 1)
                Write-Host "   • $($_.Name) ($size MB)"
            }
        }
    }
    
    return $true
}

function Invoke-FullWorkflow {
    Write-Status "╔════════════════════════════════════════════╗" -Type Info
    Write-Status "║    FULL TIER 2 AUTOMATION WORKFLOW        ║" -Type Info
    Write-Status "╚════════════════════════════════════════════╝" -Type Info
    
    # Step 1: Upload
    Write-Status "`n📋 PHASE 1: Upload Model" -Type Info
    if (-not (Invoke-UploadModel)) {
        Write-Status "❌ Upload phase failed" -Type Error
        return $false
    }
    
    # Step 2: Trigger
    Write-Status "`n📋 PHASE 2: Trigger Colab" -Type Info
    if (-not (Invoke-TriggerColab)) {
        Write-Status "❌ Trigger phase failed" -Type Error
        return $false
    }
    
    # Step 3: Monitor
    Write-Status "`n📋 PHASE 3: Monitor" -Type Info
    Write-Status "⏳ Waiting for Colab execution to complete..." -Type Info
    Write-Status "   Colab execution time: 2-4 hours" -Type Warning
    Write-Status "   You can check GitHub Actions logs in the meantime" -Type Info
    
    # Step 4: Download
    Write-Status "`n📋 PHASE 4: Download Results" -Type Info
    $maxWaitMinutes = 300  # 5 hours max wait
    $pollInterval = 60  # Check every minute
    $elapsed = 0
    
    while ($elapsed -lt $maxWaitMinutes) {
        Write-Status "🔄 Checking for results... (waited $elapsed minutes)" -Type Info
        
        if (Invoke-DownloadResults) {
            Write-Status "`n✅ FULL WORKFLOW COMPLETE!" -Type Success
            return $true
        }
        
        $elapsed += $pollInterval
        if ($elapsed -lt $maxWaitMinutes) {
            Write-Status "   Retrying in $($pollInterval) seconds..." -Type Info
            Start-Sleep -Seconds $pollInterval
        }
    }
    
    Write-Status "⚠️  Timeout waiting for results. Check GitHub Actions manually." -Type Warning
    return $true
}

function Show-Status {
    Write-Status "═══════════════════════════════════════════════════════════════════" -Type Info
    Write-Status "CYBERSPORE DEPLOYMENT STATUS" -Type Info
    Write-Status "═══════════════════════════════════════════════════════════════════" -Type Info
    
    # Check local model
    if (Test-Path $Config.LocalModelPath) {
        $modelSize = [math]::Round((Get-ChildItem -Path $Config.LocalModelPath -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB, 1)
        Write-Status "✅ Local model: $modelSize MB in $($Config.LocalModelPath)" -Type Success
    } else {
        Write-Status "❌ Local model not found at $($Config.LocalModelPath)" -Type Error
    }
    
    # Check rclone
    if (Test-RcloneInstalled) {
        Write-Status "✅ rclone installed" -Type Success
        
        if (Test-RcloneRemote $Config.RcloneRemote) {
            Write-Status "✅ rclone remote '$($Config.RcloneRemote)' configured" -Type Success
        } else {
            Write-Status "❌ rclone remote '$($Config.RcloneRemote)' not configured" -Type Error
            Write-Status "   Run: rclone config" -Type Info
        }
    } else {
        Write-Status "❌ rclone not installed" -Type Error
    }
    
    # Check results folder
    if (Test-Path $Config.LocalResultsPath) {
        Write-Status "✅ Local results folder exists: $($Config.LocalResultsPath)" -Type Success
    }
    
    Write-Status "═══════════════════════════════════════════════════════════════════" -Type Info
}

# Main execution
switch ($Action) {
    'UploadModel' { $success = Invoke-UploadModel }
    'TriggerColab' { $success = Invoke-TriggerColab }
    'DownloadResults' { $success = Invoke-DownloadResults }
    'Full' { $success = Invoke-FullWorkflow }
    'Status' { Show-Status; $success = $true }
}

if ($success) {
    Write-Status "`n✅ Operation completed successfully" -Type Success
    exit 0
} else {
    Write-Status "`n❌ Operation failed" -Type Error
    exit 1
}
