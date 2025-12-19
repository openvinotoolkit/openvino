# Tier 2 Automation Setup Guide

## Overview

This guide walks you through setting up **Tier 2 Automation** for Cyberspore:
- **File sync**: Upload infected model to Google Drive with `rclone`
- **Notebook execution**: Trigger Colab with GitHub Actions (manual execution with full automation framework)
- **Result collection**: Download results automatically

**Total setup time**: ~30 minutes (mostly waiting for downloads)

---

## Prerequisites

- Windows PowerShell 5.1+ or PowerShell Core
- Python 3.10+
- GitHub account with write access to repository
- Google account with Google Drive access

---

## Step 1: Install rclone

rclone is the core tool for syncing files to Google Drive.

### Option A: Download (Recommended)

1. **Download rclone**
   - Go to https://rclone.org/downloads/
   - Download: `rclone-v1.68-windows-amd64.zip`
   - Extract to: `C:\Program Files\rclone\` (or similar)

2. **Add to PATH**
   - Open: Settings → System → Environment Variables
   - Click: "Edit the system environment variables"
   - New Variable Name: `PATH`
   - Add value: `C:\Program Files\rclone`
   - Restart PowerShell

3. **Verify installation**
   ```powershell
   rclone --version
   # Should output: rclone v1.68.0
   ```

### Option B: Package Manager

```powershell
# Using Winget (if available)
winget install rclone

# Using Chocolatey
choco install rclone
```

---

## Step 2: Configure rclone for Google Drive

### Interactive Configuration

1. **Start configuration**
   ```powershell
   rclone config
   ```

2. **Follow prompts**
   - Choose: `n` (new remote)
   - Name: `gdrive` (or your preference)
   - Storage type: `drive` (Google Drive)
   - Client ID: Press Enter (use default)
   - Client Secret: Press Enter (use default)
   - Scope: `1` (Full access)
   - Service account: `n` (use personal account)
   - Auto-config: `y` (opens browser for auth)

3. **Complete OAuth flow**
   - Browser opens asking for Google permission
   - Click "Allow" to grant Drive access
   - Confirm in terminal

4. **Verify configuration**
   ```powershell
   rclone ls gdrive:
   # Should list files/folders in your Drive
   ```

### Configuration File Location
- Windows: `%APPDATA%\rclone\rclone.conf`
- Contains encrypted credentials (safe to commit to git)

---

## Step 3: Set Up GitHub Secrets

GitHub Actions needs credentials to authenticate with Google Drive (optional, but recommended for full automation).

### Required Secrets

1. **Go to GitHub**
   - Repository → Settings → Secrets and Variables → Actions

2. **Add these secrets:**

   **GOOGLE_DRIVE_FOLDER_ID**
   - Get this from your Google Drive folder URL
   - URL format: `https://drive.google.com/drive/folders/FOLDER_ID`
   - Example: `1A_bC2dE3fG4hI5jK6lM7nO8pQ9rS0tU`
   - Click: "New repository secret"
   - Name: `GOOGLE_DRIVE_FOLDER_ID`
   - Value: Your folder ID

   **COLAB_NOTEBOOK_ID** (optional)
   - Get from Colab share link
   - Format: `notebooks/NOTEBOOK_ID`

   **GOOGLE_DRIVE_API_CREDENTIALS** (optional, for advanced automation)
   - Not required for basic tier 2

---

## Step 4: Test rclone Sync

Before running the full automation, test the sync locally.

```powershell
# List your Drive root
rclone ls gdrive:

# Create test folder
rclone mkdir gdrive:Cyberspore

# List the folder (should be empty)
rclone ls gdrive:Cyberspore

# Dry-run sync (shows what would be uploaded)
rclone sync gemma_ir_tssn gdrive:Cyberspore/gemma_ir_tssn --dry-run

# Actual sync (uploads the model)
rclone sync gemma_ir_tssn gdrive:Cyberspore/gemma_ir_tssn --progress
```

**Expected output:**
```
Transferred:        600.2 MB / 600.2 MB, 100%, 2.5 MB/s, ETA 0s
```

---

## Step 5: Set Up Automation Scripts

The automation includes 3 key scripts:

### A. `deploy_to_colab.ps1` - Main Orchestrator

**Usage:**
```powershell
# Upload model only
.\deploy_to_colab.ps1 -Action UploadModel

# Full automation (upload → trigger → wait → download)
.\deploy_to_colab.ps1 -Action Full

# Trigger Colab manually
.\deploy_to_colab.ps1 -Action TriggerColab

# Download results only
.\deploy_to_colab.ps1 -Action DownloadResults

# Check status
.\deploy_to_colab.ps1 -Action Status

# With custom settings
.\deploy_to_colab.ps1 -Action UploadModel -RcloneRemote gdrive -DriveFolderPath MyFolder
```

### B. `download_results.py` - Result Downloader

**Usage:**
```bash
python download_results.py

# With custom paths
python download_results.py --remote gdrive --drive-path Cyberspore/results --local-path ./my_results

# List files without downloading
python download_results.py --list-only

# Force gdown instead of rclone
python download_results.py --method gdown
```

### C. GitHub Actions Workflow

File: `.github/workflows/colab-evolution.yml`

**Trigger:**
- GitHub Actions tab → colab-evolution workflow → "Run workflow" button
- Or via GitHub CLI: `gh workflow run colab-evolution.yml`

---

## Step 6: Run Full Tier 2 Workflow

### Complete Automated Flow

```powershell
# 1. Upload model (5-15 minutes depending on internet)
.\deploy_to_colab.ps1 -Action UploadModel

# Output:
# [14:27:45] ⬆️  Uploading model (this may take 5-15 minutes for 600MB)...
# [14:42:31] ✅ Model uploaded successfully in 15.1 minutes
```

### Then (Manual Step) - Run Colab Notebook

Since there's no direct Colab execution API, you must manually run the notebook:

1. **Go to**: https://colab.research.google.com
2. **Open**: `Cyberspore_Evolution_Remote.ipynb`
   - Upload from: `c:\Users\ssdaj\openvino\Cyberspore_Evolution_Remote.ipynb`
3. **Set Runtime**: Runtime → Change Runtime Type → GPU (T4)
4. **Run Cells** (sequentially):
   - Cell 1: Mount Google Drive
   - Cell 2: Clone repository
   - Cell 3: Install dependencies (~2 min)
   - Cell 4: Build C++ extension (~2 min)
   - Cell 5: Load infected model from Drive
   - **Cell 6**: ⏳ Run evolution (2-4 hours)
   - Cell 7: Save results to Drive

### After Colab Completes - Download Results

```powershell
# Download results automatically
.\deploy_to_colab.ps1 -Action DownloadResults

# Or use Python directly
python download_results.py

# Check what was downloaded
ls ./colab_results/

# View the results report
cat ./colab_results/RESULTS_REPORT.md
```

---

## Architecture Diagram

```
Local Machine (Windows)
    ├── gemma_ir_tssn/ (600 MB infected model)
    └── deploy_to_colab.ps1
        │
        ├─ rclone sync ──→ Google Drive
        │                   └── /Cyberspore/gemma_ir_tssn/
        │
        ├─ GitHub Actions trigger
        │   └── (manual Colab execution for now)
        │
        └─ download_results.py ←── Google Drive
                                    └── /Cyberspore/results/
                                        ├── evolution_results.json
                                        ├── evolved_checkpoint.bin
                                        └── evolution_progress.log
```

---

## Troubleshooting

### Problem: `rclone: command not found`

**Solution:**
```powershell
# Check PATH
$env:PATH -split ';' | Select-String rclone

# Add to PATH temporarily
$env:PATH += ';C:\Program Files\rclone'

# Verify
rclone --version
```

### Problem: Google Drive auth fails

**Solution:**
```powershell
# Reconfigure rclone
rclone config

# Edit existing 'gdrive' remote
# Choose 'e' to edit, then 'd' to delete and recreate
```

### Problem: Upload too slow

**Solution:**
```powershell
# Increase parallel transfers
rclone sync gemma_ir_tssn gdrive:Cyberspore/gemma_ir_tssn `
  --transfers 8 `
  --checkers 8 `
  --multi-thread-streams 4
```

### Problem: Colab gives "model not found"

**Solution:**
1. Verify model uploaded: `rclone ls gdrive:Cyberspore/gemma_ir_tssn`
2. In Colab Cell 6, check Drive mounting worked: `!ls /content/drive/MyDrive/Cyberspore/gemma_ir_tssn/`
3. Manually copy if needed: `!cp -r /content/drive/MyDrive/Cyberspore/gemma_ir_tssn .`

---

## Advanced: Parallel Evolution Jobs

Run multiple evolution experiments simultaneously with different hyperparameters:

```powershell
# Job 1: Standard hyperparameters
.\deploy_to_colab.ps1 -Action UploadModel -DriveFolderPath Cyberspore_Job1

# Job 2: Different learning rate
.\deploy_to_colab.ps1 -Action UploadModel -DriveFolderPath Cyberspore_Job2

# In Colab, customize cells 6-7 with different hyperparameters
# Results save to: Cyberspore_Job1/results/ and Cyberspore_Job2/results/

# Download all
python download_results.py --drive-path Cyberspore_Job1/results --local-path ./job1_results
python download_results.py --drive-path Cyberspore_Job2/results --local-path ./job2_results
```

---

## Next Steps

1. **Install rclone** (Step 1)
2. **Configure Google Drive** (Step 2)
3. **Test sync** (Step 4)
4. **Run first upload** (Step 6)
5. **Execute Colab manually** (Step 6)
6. **Download results** (Step 6)

---

## Future Enhancements (Tier 3)

Once Tier 2 is stable, we can add:
- Google Cloud Functions trigger for auto-execution
- Selenium-based Colab automation (fragile, use sparingly)
- Slack/Discord notifications for completion
- Result analysis and visualization pipeline
- Cost tracking for Colab GPU usage

---

## Support

For issues, check:
- rclone docs: https://rclone.org/
- Google Drive API: https://developers.google.com/drive
- Colab docs: https://colab.research.google.com/

