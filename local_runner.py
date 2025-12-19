#!/usr/bin/env python3
"""
LOCAL INFECTION RUNNER
======================
Run Cyberspore infection locally in VS Code.
This is the fast part—just injects TSSN nodes into Gemma.
Does NOT require Colab.
"""

import sys
import os
from pathlib import Path
import subprocess
import platform

def setup_environment():
    """Setup local environment."""
    cwd = Path.cwd()
    is_windows = platform.system() == "Windows"
    
    print("\ud83d\udd27 Cyberspore Local Infection Runner")
    print("=" * 50)
    print(f"\ud83d\udcbb OS: {platform.system()}")
    print(f"\ud83d\udccb Working Directory: {cwd}")
    
    if is_windows:
        release_bin = cwd / 'bin/intel64/Release'
        tbb_bin = cwd / 'temp/Windows_AMD64/tbb/bin'
        
        if release_bin.exists() and tbb_bin.exists():
            os.environ['OPENVINO_LIB_PATHS'] = f"{release_bin.absolute()};{tbb_bin.absolute()}"
            print(f"\u2705 OpenVINO environment configured")
        else:
            print(f"\u26a0\ufe0f OpenVINO build not found. Using system installation.")
    
    return cwd

def check_gemma_model(cwd):
    """Verify Gemma IR exists."""
    gemma_path = cwd / "gemma_ir" / "openvino_model.xml"
    
    if not gemma_path.exists():
        print(f"\n\u274c ERROR: Gemma model not found at {gemma_path}")
        print(f"\n\ud83d\udce1 TIP: You need to convert Gemma-2B to OpenVINO format first.")
        print(f"   Run: optimum-cli export openvino --model google/gemma-2b gemma_ir/")
        return False
    
    print(f"\u2705 Gemma model found: {gemma_path}")
    return True

def run_infection():
    """Execute the infection script."""
    print(f"\n\ud83d\udc4b Running infection...")
    print("   Injecting TSSN layers into Gemma...")
    
    result = subprocess.run(
        [sys.executable, "infect_gemma_test.py"],
        capture_output=False
    )
    
    if result.returncode == 0:
        print(f"\n\u2705 Infection complete!")
        print(f"   Output: gemma_ir_tssn/openvino_model.xml")
        return True
    else:
        print(f"\n\u274c Infection failed!")
        return False

def main():
    cwd = setup_environment()
    
    # Check prerequisites
    if not check_gemma_model(cwd):
        return 1
    
    # Run infection
    if not run_infection():
        return 1
    
    print(f"\n\ud83c\udf89 Success!")
    print(f"   Next step: Submit evolution job to Colab")
    print(f"   Run: python submit_to_colab.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
