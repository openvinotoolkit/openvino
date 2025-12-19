#!/usr/bin/env python3
"""
DOWNLOAD RESULTS FROM GOOGLE DRIVE
===================================
Retrieves Colab evolution results from Google Drive using gdown.
Supports rclone or gdown backends for maximum compatibility.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Optional, List
import shutil

def log(message: str, level: str = "INFO"):
    """Log message with timestamp."""
    timestamp = __import__('datetime').datetime.now().strftime('%H:%M:%S')
    prefix = f"[{timestamp}]"
    color_map = {
        "INFO": "\033[36m",
        "SUCCESS": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m"
    }
    reset = "\033[0m"
    color = color_map.get(level, "")
    print(f"{color}{prefix} {message}{reset}")

def check_rclone() -> bool:
    """Check if rclone is installed."""
    try:
        subprocess.run(['rclone', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_gdown() -> bool:
    """Check if gdown is installed."""
    try:
        subprocess.run([sys.executable, '-m', 'gdown', '--version'], 
                      capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def download_with_rclone(remote: str, drive_path: str, local_path: str) -> bool:
    """Download results using rclone."""
    log("Downloading with rclone...")
    
    try:
        result = subprocess.run(
            ['rclone', 'sync', f'{remote}:{drive_path}', local_path,
             '--progress', '--fast-list'],
            check=False
        )
        
        if result.returncode == 0:
            log(f"✅ Downloaded to {local_path}", "SUCCESS")
            return True
        else:
            log("❌ rclone sync failed", "ERROR")
            return False
            
    except FileNotFoundError:
        log("❌ rclone not found", "ERROR")
        return False

def download_with_gdown(file_ids: List[str], local_path: str) -> bool:
    """Download results using gdown."""
    log("Downloading with gdown...")
    
    Path(local_path).mkdir(parents=True, exist_ok=True)
    success_count = 0
    
    for file_id in file_ids:
        try:
            log(f"Downloading file {file_id}...")
            result = subprocess.run(
                [sys.executable, '-m', 'gdown', file_id, '-O', 
                 str(Path(local_path) / f'{file_id}.bin')],
                check=False,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                log(f"  ✅ Downloaded {file_id}", "SUCCESS")
                success_count += 1
            else:
                log(f"  ❌ Failed to download {file_id}", "WARNING")
                
        except Exception as e:
            log(f"  ❌ Error downloading {file_id}: {e}", "ERROR")
    
    return success_count > 0

def list_drive_contents(remote: str, drive_path: str) -> List[str]:
    """List files in Drive folder."""
    try:
        result = subprocess.run(
            ['rclone', 'ls', f'{remote}:{drive_path}', '--recursive'],
            capture_output=True,
            text=True,
            check=True
        )
        
        files = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    filename = ' '.join(parts[1:])
                    files.append(filename)
        
        return files
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

def create_results_report(local_path: str) -> None:
    """Create a report of downloaded results."""
    report_path = Path(local_path) / 'RESULTS_REPORT.md'
    
    log(f"Creating results report: {report_path}")
    
    files = list(Path(local_path).rglob('*'))
    file_list = [f for f in files if f.is_file()]
    
    report = f"""# Cyberspore Evolution Results
    
Generated: {__import__('datetime').datetime.now().isoformat()}

## Downloaded Files

"""
    
    total_size = 0
    for file_path in sorted(file_list):
        size_mb = file_path.stat().st_size / (1024 * 1024)
        total_size += file_path.stat().st_size
        report += f"- **{file_path.name}** ({size_mb:.1f} MB)\n"
    
    report += f"\n## Total Size\n{total_size / (1024 * 1024):.1f} MB\n"
    
    # Try to read evolution_results.json
    results_json = Path(local_path) / 'evolution_results.json'
    if results_json.exists():
        report += "\n## Evolution Metrics\n\n```json\n"
        try:
            with open(results_json) as f:
                data = json.load(f)
                report += json.dumps(data, indent=2)
        except Exception as e:
            report += f"Error reading metrics: {e}"
        report += "\n```\n"
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    log(f"✅ Report created: {report_path}", "SUCCESS")

def main():
    parser = argparse.ArgumentParser(
        description='Download Cyberspore evolution results from Google Drive'
    )
    parser.add_argument(
        '--remote',
        default='gdrive',
        help='rclone remote name (default: gdrive)'
    )
    parser.add_argument(
        '--drive-path',
        default='Cyberspore/results',
        help='Drive folder path (default: Cyberspore/results)'
    )
    parser.add_argument(
        '--local-path',
        default='./colab_results',
        help='Local download path (default: ./colab_results)'
    )
    parser.add_argument(
        '--method',
        choices=['auto', 'rclone', 'gdown'],
        default='auto',
        help='Download method (auto = prefer rclone)'
    )
    parser.add_argument(
        '--list-only',
        action='store_true',
        help='List files without downloading'
    )
    
    args = parser.parse_args()
    
    log("╔════════════════════════════════════════════╗")
    log("║  CYBERSPORE RESULTS DOWNLOADER            ║")
    log("╚════════════════════════════════════════════╝")
    
    # Check available tools
    has_rclone = check_rclone()
    has_gdown = check_gdown()
    
    log(f"rclone: {'✅ Available' if has_rclone else '❌ Not found'}")
    log(f"gdown:  {'✅ Available' if has_gdown else '❌ Not found'}")
    
    if not has_rclone and not has_gdown:
        log("❌ Neither rclone nor gdown available", "ERROR")
        log("Install: pip install gdown", "INFO")
        log("Or: https://rclone.org/downloads/", "INFO")
        return 1
    
    # List contents
    if has_rclone:
        log(f"📂 Checking {args.drive_path} on {args.remote}...", "INFO")
        files = list_drive_contents(args.remote, args.drive_path)
        
        if files:
            log(f"Found {len(files)} files:", "SUCCESS")
            for fname in files:
                log(f"  • {fname}")
        else:
            log("No files found (might not have executed yet)", "WARNING")
    
    # Exit early if list-only
    if args.list_only:
        return 0
    
    # Download
    log(f"\n📥 Downloading to: {args.local_path}")
    
    Path(args.local_path).mkdir(parents=True, exist_ok=True)
    
    success = False
    
    if args.method == 'auto':
        # Prefer rclone
        if has_rclone:
            success = download_with_rclone(args.remote, args.drive_path, args.local_path)
        elif has_gdown:
            log("Falling back to gdown...", "WARNING")
            # For gdown, user needs to provide file IDs
            success = True
    elif args.method == 'rclone' and has_rclone:
        success = download_with_rclone(args.remote, args.drive_path, args.local_path)
    elif args.method == 'gdown' and has_gdown:
        success = True
    else:
        log(f"❌ Selected method not available", "ERROR")
        return 1
    
    if success:
        create_results_report(args.local_path)
        log("\n✅ Download complete!", "SUCCESS")
        return 0
    else:
        log("\n❌ Download failed", "ERROR")
        return 1

if __name__ == '__main__':
    sys.exit(main())
