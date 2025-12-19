#!/usr/bin/env python3
"""
COLAB AUTOMATION SETUP SCRIPT
============================
Initializes rclone, GitHub Actions secrets, and deployment infrastructure.
Run this ONCE to set up the automation environment.
"""

import subprocess
import sys
import os
from pathlib import Path
import json

def check_rclone():
    """Check if rclone is installed."""
    try:
        result = subprocess.run(['rclone', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ rclone is installed")
            return True
    except FileNotFoundError:
        pass
    return False

def install_rclone():
    """Provide rclone installation instructions."""
    print("\n📦 Installing rclone...")
    print("=" * 60)
    print("\nOption 1: Download (Recommended for Windows)")
    print("  1. Go to: https://rclone.org/downloads/")
    print("  2. Download: rclone-v1.68-windows-amd64.zip")
    print("  3. Extract and add to PATH")
    print("  4. Verify: rclone --version")
    print("\nOption 2: Winget (if you have it)")
    print("  Run: winget install rclone")
    print("\n" + "=" * 60)
    return False

def setup_rclone_config():
    """Generate rclone config instructions."""
    config_path = Path.home() / '.config' / 'rclone' / 'rclone.conf'
    
    print("\n🔑 Setting up rclone Google Drive authentication...")
    print("=" * 60)
    print("\nRun this command in your terminal:")
    print("  rclone config")
    print("\nThen follow these steps:")
    print("  1. Select 'n' for new remote")
    print("  2. Name: 'gdrive' (or your preference)")
    print("  3. Storage type: 'drive' (Google Drive)")
    print("  4. Client ID: [press Enter for default]")
    print("  5. Client Secret: [press Enter for default]")
    print("  6. Scope: '1' (drive)")
    print("  7. Service Account: 'n' (use personal account)")
    print("  8. Auto-config: 'y' (opens browser for auth)")
    print("  9. Complete OAuth flow in browser")
    print("  10. Verify: 'y'")
    print("\n✅ Config saved to:", config_path)
    print("=" * 60)

def create_deployment_config():
    """Create deployment configuration file."""
    config = {
        "rclone": {
            "remote_name": "gdrive",
            "drive_folder": "Cyberspore",
            "model_subfolder": "gemma_ir_tssn",
            "results_subfolder": "results"
        },
        "github": {
            "repo": "ssdajoker/openvino",
            "workflow": "colab-evolution.yml",
            "secrets_needed": [
                "GOOGLE_DRIVE_FOLDER_ID",
                "COLAB_NOTEBOOK_ID",
                "DRIVE_API_KEY"
            ]
        },
        "colab": {
            "notebook": "Cyberspore_Evolution_Remote.ipynb",
            "setup_time_minutes": 7,
            "evolution_time_hours": "2-4",
            "expected_output_files": [
                "evolution_results.json",
                "evolution_progress.log",
                "evolved_checkpoint.bin"
            ]
        }
    }
    
    config_path = Path('automation_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Deployment config created: {config_path}")
    return config

def print_next_steps():
    """Print implementation steps."""
    print("\n" + "=" * 70)
    print("TIER 2 AUTOMATION SETUP - NEXT STEPS")
    print("=" * 70)
    
    steps = [
        ("1. INSTALL RCLONE", [
            "Download from: https://rclone.org/downloads/",
            "Extract and add to PATH",
            "Verify: rclone --version"
        ]),
        ("2. CONFIGURE RCLONE", [
            "Run: rclone config",
            "Create 'gdrive' remote with Google Drive auth",
            "Test: rclone ls gdrive:"
        ]),
        ("3. CREATE GITHUB SECRETS", [
            "Repo Settings → Secrets and Variables → Actions",
            "Add GOOGLE_DRIVE_FOLDER_ID (from Drive URL)",
            "Add COLAB_NOTEBOOK_ID (from Colab share link)",
            "Add DRIVE_API_KEY (optional, for enhanced auth)"
        ]),
        ("4. UPLOAD INITIAL MODEL", [
            "Run: python deploy_to_colab.ps1 -Action UploadModel",
            "Syncs gemma_ir_tssn/ to /Cyberspore/gemma_ir_tssn/"
        ]),
        ("5. TRIGGER COLAB", [
            "GitHub Actions → colab-evolution → Run workflow",
            "Or wait for auto-trigger on model upload"
        ]),
        ("6. MONITOR EXECUTION", [
            "Check GitHub Actions logs in real-time",
            "Check /Cyberspore/results/ on Drive for outputs"
        ]),
        ("7. DOWNLOAD RESULTS", [
            "Run: python deploy_to_colab.ps1 -Action DownloadResults",
            "Results saved to ./colab_results/"
        ])
    ]
    
    for title, items in steps:
        print(f"\n{title}")
        print("-" * 70)
        for item in items:
            print(f"  • {item}")
    
    print("\n" + "=" * 70)
    print("FILES TO CREATE NEXT:")
    print("=" * 70)
    print("  1. deploy_to_colab.ps1 ......... PowerShell deployment wrapper")
    print("  2. .github/workflows/colab-evolution.yml ... GitHub Actions trigger")
    print("  3. download_results.py ........ Result downloader script")
    print("  4. automation_config.json .... Deployment configuration")
    print("=" * 70)

def main():
    print("\n" + "=" * 70)
    print("CYBERSPORE TIER 2 AUTOMATION SETUP")
    print("=" * 70)
    
    # Check rclone
    if not check_rclone():
        print("\n⚠️  rclone not found. Install it first:")
        install_rclone()
        print("\nOnce installed, run this script again.")
        return 1
    
    # Create config
    config = create_deployment_config()
    
    # Setup instructions
    setup_rclone_config()
    
    # Print next steps
    print_next_steps()
    
    print("\n✅ Setup initialization complete!")
    print("\nContinue with file creation steps above.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
