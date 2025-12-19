#!/usr/bin/env python3
"""
COLAB JOB SUBMISSION SCRIPT
===========================
Run this from VS Code to submit the evolution job to Colab.
Sets up Google Drive, uploads infected model, runs evolution remotely.
"""

import os
import sys
import json
from pathlib import Path
import subprocess

def submit_to_colab():
    \"\"\"Create and submit a Colab notebook with the evolution job.\"\"\"
    
    print("\ud83d\udce4 Colab Job Submission")
    print("=" * 50)
    
    # Check if infected model exists
    infected_model = Path("gemma_ir_tssn/openvino_model.xml")
    if not infected_model.exists():
        print("\u274c Error: Infected model not found!")
        print("   Run: python local_runner.py")
        return False
    
    print(f"\u2705 Infected model ready: {infected_model}")
    
    # Create Colab notebook content
    colab_notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Cyberspore Evolution - Remote Job\\n",
                    "\\n",
                    "This notebook runs the heavy evolution computation on Colab.\\n",
                    "Infected model is uploaded from your local machine."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Mount Google Drive\\n",
                    "from google.colab import drive\\n",
                    "drive.mount('/content/drive')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Clone repo\\n",
                    "!git clone https://github.com/ssdajoker/openvino.git\\n",
                    "%cd openvino"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Install dependencies\\n",
                    "!pip install -q openvino-dev numpy pandas tqdm torch transformers"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Build C++ extension\\n",
                    "import os\\n",
                    "os.makedirs('src/custom_ops/build', exist_ok=True)\\n",
                    "!cd src/custom_ops/build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j4"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Download infected model from Drive (or upload if first time)\\n",
                    "import shutil\\n",
                    "from google.colab import files\\n",
                    "\\n",
                    "# If you're running this for the first time, upload the infected model\\n",
                    "print('Option 1: Upload infected model (first time only)')\\n",
                    "print('Option 2: Copy from Drive (if already uploaded)')\\n",
                    "\\n",
                    "# For now, assume it's on Drive:\\n",
                    "!cp -r /content/drive/MyDrive/Cyberspore/gemma_ir_tssn . || echo 'Not on Drive yet'"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Run evolution (the heavy computation)\\n",
                    "!python evolve_gemma_v4_steady_state.py"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Save results back to Drive\\n",
                    "!mkdir -p /content/drive/MyDrive/Cyberspore/results\\n",
                    "!cp -r gemma_ir_tssn /content/drive/MyDrive/Cyberspore/results/\\n",
                    "!cp evolution_results.json /content/drive/MyDrive/Cyberspore/results/\\n",
                    "!cp evolution_progress.log /content/drive/MyDrive/Cyberspore/results/ 2>/dev/null || echo 'No log file'\\n",
                    "\\n",
                    "print('✅ Results saved to Google Drive!')"
                ]
            }
        ],
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    # Save notebook
    notebook_path = Path("Cyberspore_Evolution_Remote.ipynb")
    with open(notebook_path, 'w') as f:
        json.dump(colab_notebook, f, indent=2)
    
    print(f"\n\u2705 Colab notebook created: {notebook_path}")
    
    # Instructions
    print("\n\ud83d\udce4 NEXT STEPS:")
    print("=" * 50)
    print("1. Go to: https://colab.research.google.com")
    print("2. Upload the notebook: Cyberspore_Evolution_Remote.ipynb")
    print("3. (First time only) Upload your gemma_ir_tssn folder to Google Drive")
    print("4. Run the notebook cells sequentially")
    print("5. Results will be saved to: /MyDrive/Cyberspore/results/")
    print("6. Download from Drive when complete")
    
    return True

if __name__ == "__main__":
    success = submit_to_colab()
    sys.exit(0 if success else 1)
