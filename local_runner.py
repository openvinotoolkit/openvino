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
import io
from pathlib import Path
import subprocess
import platform

# Fix UTF-8 encoding on Windows PowerShell for emoji support
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def setup_environment():
    """Setup local environment."""
    cwd = Path.cwd()
    is_windows = platform.system() == "Windows"
    
    print("🔧 Cyberspore Local Infection Runner")
    print("=" * 50)
    print(f"💻 OS: {platform.system()}")
    print(f"📋 Working Directory: {cwd}")
    
    if is_windows:
        release_bin = cwd / 'bin/intel64/Release'
        tbb_bin = cwd / 'temp/Windows_AMD64/tbb/bin'
        
        if release_bin.exists() and tbb_bin.exists():
            os.environ['OPENVINO_LIB_PATHS'] = f"{release_bin.absolute()};{tbb_bin.absolute()}"
            print(f"✅ OpenVINO environment configured")
        else:
            print(f"⚠️ OpenVINO build not found. Using system installation.")
    
    return cwd

def check_gemma_model(cwd):
    """Verify Gemma IR exists."""
    gemma_path = cwd / "gemma_ir" / "openvino_model.xml"
    
    if not gemma_path.exists():
        print(f"\n❌ ERROR: Gemma model not found at {gemma_path}")
        print(f"\n💡 TIP: You need to convert Gemma-2B to OpenVINO format first.")
        print(f"   Run: optimum-cli export openvino --model google/gemma-2b gemma_ir/")
        return False
    
    print(f"✅ Gemma model found: {gemma_path}")
    return True

def run_infection():
    """Execute the infection script."""
    print(f"\n👋 Running infection...")
    print("   Injecting TSSN layers into Gemma...")
    
    # Source the dev environment to ensure OpenVINO is in PYTHONPATH
    # On Windows, we need to run infect_gemma_test.py in a PowerShell that has sourced setup_dev_env.ps1
    # However, for Windows, we can also set PYTHONPATH directly here
    cwd = Path.cwd()
    python_path = cwd / 'bin/intel64/Release/python'
    
    env = os.environ.copy()
    if python_path.exists():
        # Prepend the local Python packages to PYTHONPATH
        existing_pythonpath = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = f"{python_path};{existing_pythonpath}" if existing_pythonpath else str(python_path)
    
    result = subprocess.run(
        [sys.executable, "infect_gemma_test.py"],
        capture_output=False,
        env=env
    )
    
    # Check if infected model was created (it succeeds even if embedding layer fails)
    infected_model = cwd / "gemma_ir_tssn" / "openvino_model.xml"
    if infected_model.exists():
        print(f"\n✅ Infection complete!")
        print(f"   Output: gemma_ir_tssn/openvino_model.xml")
        print(f"   Note: Embedding layer may have been skipped due to memory constraints.")
        return True
    else:
        print(f"\n❌ Infection failed!")
        return False

def main():
    cwd = setup_environment()
    
    # Check prerequisites
    if not check_gemma_model(cwd):
        return 1
    
    # Run infection
    if not run_infection():
        return 1
    
    print(f"\n🎉 Success!")
    print(f"   Next step: Submit evolution job to Colab")
    print(f"   Run: python submit_to_colab.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
