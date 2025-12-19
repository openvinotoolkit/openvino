# Tier 2 Automation Quick Start

**⏱️ Total Time**: ~30-45 minutes (mostly downloads)

## ✅ Pre-Flight Checklist

- [ ] Windows PowerShell 5.1+ installed (`pwsh -Version`)
- [ ] Python 3.10+ installed (`python --version`)
- [ ] GitHub account with write access to repository
- [ ] Google Drive access (personal or workspace account)
- [ ] Internet connection (600MB+ upload/download)
- [ ] 2-3 GB free disk space locally

---

## 🚀 Quick Setup (15 minutes)

### 1. Install rclone (5 min)
```powershell
# Download from https://rclone.org/downloads/
# Or use Winget:
winget install rclone

# Verify
rclone --version
```

### 2. Configure Google Drive (5 min)
```powershell
rclone config
# Select: n (new remote)
# Name: gdrive
# Type: drive (Google Drive)
# Accept defaults, auth in browser
```

### 3. Verify rclone works (2 min)
```powershell
rclone ls gdrive:
# Should list your Drive contents
```

### 4. Copy automation scripts (1 min)
All files are already in your repository:
- ✅ `deploy_to_colab.ps1`
- ✅ `download_results.py`
- ✅ `.github/workflows/colab-evolution.yml`

### 5. Optional: GitHub Secrets (2 min)
For full automation, add to repo Settings → Secrets:
- `GOOGLE_DRIVE_FOLDER_ID` = Your Drive folder ID
- `COLAB_NOTEBOOK_ID` = Optional

---

## 🎯 Execution Workflow (30-120 minutes total)

### Phase 1: Upload Model (5-15 min)
```powershell
.\deploy_to_colab.ps1 -Action UploadModel

# Output:
# [14:27:45] ⬆️  Uploading model...
# [14:42:31] ✅ Model uploaded successfully in 15.1 minutes
```

### Phase 2: Run Colab Manually (120-240 min)
1. Go to: https://colab.research.google.com
2. Upload: `Cyberspore_Evolution_Remote.ipynb`
3. Set Runtime → GPU (T4)
4. Run cells 1-7 sequentially
5. **Cell 6 takes 2-4 hours** ⏳

### Phase 3: Download Results (5-15 min)
```powershell
.\deploy_to_colab.ps1 -Action DownloadResults

# Or use Python:
python download_results.py

# Results in: ./colab_results/
```

---

## 📊 Status Checking

Check current state anytime:
```powershell
.\deploy_to_colab.ps1 -Action Status

# Output:
# ✅ Local model: 600.2 MB
# ✅ rclone installed
# ✅ rclone remote 'gdrive' configured
# ✅ Local results folder exists
```

---

## 🔍 Verification Points

After each phase, verify:

**After Upload:**
```powershell
rclone ls gdrive:Cyberspore/gemma_ir_tssn/

# Should show:
# openvino_model.xml (2.9 MB)
# openvino_model.bin (600.2 MB)
```

**During Colab:**
- ✅ Cell 1: Drive mounted
- ✅ Cell 2: Repo cloned
- ✅ Cell 3-4: Dependencies & build OK
- ✅ Cell 5: Model loaded
- ⏳ Cell 6: Evolution running (monitor output)
- ✅ Cell 7: Results saved to Drive

**After Download:**
```powershell
ls ./colab_results/

# Should show:
# evolution_results.json
# evolved_checkpoint.bin
# evolution_progress.log (if available)
```

---

## 🆘 Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| `rclone: command not found` | Add to PATH: `C:\Program Files\rclone` |
| `Google auth fails` | Run `rclone config` to reconfigure |
| `Upload slow` | Use `-transfers 8 -checkers 8` in deploy script |
| `Colab can't find model` | Check Drive: `!ls /content/drive/MyDrive/Cyberspore/` |
| `Results not appearing` | Check Drive manually: Cyberspore/results/ |

---

## 📞 Support Resources

| Tool | Docs | Issue |
|------|------|-------|
| **rclone** | https://rclone.org/ | `rclone config` |
| **Google Drive API** | https://developers.google.com/drive | Credentials |
| **Colab** | https://colab.research.google.com/ | GPU access |
| **GitHub Actions** | https://docs.github.com/actions | Workflow auth |

---

## ⏭️ What's Next After Results

Once you have `evolution_results.json`:

1. **Analyze metrics**
   ```powershell
   cat ./colab_results/evolution_results.json | python -m json.tool
   ```

2. **Evaluate model quality**
   - Test perplexity on benchmarks
   - Compare with baseline Gemma-2B
   - Validate inference speedup

3. **Iterate hyperparameters**
   - Modify evolution parameters
   - Re-run (create new Colab notebooks with different settings)
   - Compare results

4. **Deploy sparse model**
   - Use evolved model for inference
   - Test on Intel UHD 620 or other hardware
   - Measure actual speedup

---

## 💾 Save Progress

After successful execution, commit to git:

```powershell
git add colab_results/
git commit -m "Add evolution results from Colab run"
git push
```

---

## 🎓 Learning Path

**If you're new to this workflow:**
1. Read `TIER2_SETUP_GUIDE.md` (full details)
2. Follow this checklist step-by-step
3. Ask questions about each phase
4. Run once manually to understand flow
5. Automate after first success

---

**Status**: ✅ Ready to start!  
**Estimated total time**: ~2.5-4.5 hours (mostly Colab compute time)

Next: Go to **Step 1** above or read `TIER2_SETUP_GUIDE.md`
