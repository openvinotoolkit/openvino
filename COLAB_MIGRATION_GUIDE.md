# Project CYBERSPORE - Google Colab Setup Guide

**Objective**: Migrate the Cyberspore evolution pipeline from a local Windows machine (8GB RAM, Intel UHD 620) to Google Colab (Free T4 GPU, 12GB RAM, High-Speed CPU).

---

## 1. Why Migrate to Colab?

Your local machine is being crushed by:
- Windows background processes consuming most of your 8GB RAM.
- Slow CPU inference with the OpenVINO custom C++ extensions.
- Limited ability to run the "Metabolic War" (evolution loop) which requires iterating through thousands of candidate logic functions.

**Colab Benefits**:
- **12GB+ RAM** (enough to keep the Gemma model and candidates in memory).
- **Free GPU** (T4/A100) - We'll adapt to use CPU for now since OpenVINO's GPU plugin is Intel-specific.
- **No OS bloat** - Linux container with minimal background processes.
- **Persistent Storage** via Google Drive (save evolution checkpoints).

---

## 2. Setup Steps

### Step 1: Create a New Colab Notebook
1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. Create a new notebook.
3. (Optional) Mount your Google Drive to save outputs:
   `python
   from google.colab import drive
   drive.mount('/content/drive')
   `

### Step 2: Clone Your Repository
`python
!git clone https://github.com/ssdajoker/openvino.git
%cd openvino
`

### Step 3: Install Dependencies
`python
# OpenVINO Development (includes C++ headers for building extensions)
!pip install openvino-dev

# Python Dependencies
!pip install numpy pandas tqdm matplotlib seaborn tabulate torch transformers optimum-intel
`

### Step 4: Build the Custom C++ Extension
Your Cyberspore logic (CompositeTSSN, TSSNOp) is implemented in C++ for speed. We need to compile it.

`python
import os

# Create build directory
os.makedirs("src/custom_ops/build", exist_ok=True)

# Configure CMake
!cd src/custom_ops/build && cmake -DCMAKE_BUILD_TYPE=Release ..

# Build (use all CPU cores)
!cd src/custom_ops/build && make -jmaster(nproc)

# Verify the .so file exists
!ls -lh src/custom_ops/build/libopenvino_tssn_extension.so
`

**Expected Output**: You should see a file like libopenvino_tssn_extension.so (~1MB).

---

## 3. Adapt Your Scripts for Colab

### A. Update `infect_gemma_test.py`
This script injects TSSN nodes into the Gemma model.

**Changes Needed**:
1. **Extension Path**: Replace .dll with .so
2. **Remove Windows-specific environment setup**

`python
# OLD (Windows):
EXTENSION_PATH = cwd / "src/custom_ops/build/Release/openvino_tssn_extension.dll"

# NEW (Colab/Linux):
EXTENSION_PATH = cwd / "src/custom_ops/build/libopenvino_tssn_extension.so"
`

Remove or comment out the Windows environment setup:
`python
# REMOVE THESE LINES:
# if release_bin.exists() and tbb_bin.exists():
#     os.environ['OPENVINO_LIB_PATHS'] = f"{release_bin.absolute()};{tbb_bin.absolute()}"
# if local_python_pkg.exists():
#     sys.path.insert(0, str(local_python_pkg.absolute()))
`

### B. Update `evolve_gemma_v4_steady_state.py`
This script runs the "Metabolic War" evolution loop.

**Changes**:
1. **Device**: Change from GPU to CPU (Colab GPUs are NVIDIA, OpenVINO GPU plugin is Intel-only).
2. **Extension Path**: Same .so change as above.

`python
# OLD:
DEVICE = "GPU"
EXTENSION_PATH = Path("src/custom_ops/build/Release/openvino_tssn_extension.dll")

# NEW:
DEVICE = "CPU"
EXTENSION_PATH = Path("src/custom_ops/build/libopenvino_tssn_extension.so")
`

3. **Remove Windows Path Logic**:
`python
# REMOVE:
# sys.path.insert(0, str(local_python_pkg.absolute()))
# os.add_dll_directory(str(release_bin))
`

---

## 4. Running the Pipeline

### Step A: Infect the Model
`python
!python infect_gemma_test.py
`
**Output**: gemma_ir_tssn/openvino_model.xml (Gemma with TSSN placeholders).

### Step B: Start the Evolution
`python
!python evolve_gemma_v4_steady_state.py
`
**Output**: The script will iterate through BTL functions, trying to stabilize the model. This may take hours.

### Step C: Monitor Progress
The script should print MSE (Mean Squared Error) every few iterations:
`
Generation 10: MSE = 0.0023 | Best Function: [142, 233]
Generation 20: MSE = 0.0012 | ...
`

---

## 5. Saving Your Work

### Option A: Download Results
`python
from google.colab import files

# Download the evolved model
files.download('gemma_ir_tssn/openvino_model.xml')
files.download('gemma_ir_tssn/openvino_model.bin')
`

### Option B: Save to Google Drive
If you mounted Drive earlier:
`python
!cp -r gemma_ir_tssn /content/drive/MyDrive/Cyberspore/evolved_models/
!cp evolution_results.json /content/drive/MyDrive/Cyberspore/
`

---

## 6. Expected Performance

**Your Local Machine**:
- Inference: ~1-5 FPS (CPU-bound)
- Evolution: ~1 candidate per minute (slow)

**Colab (T4 CPU)**:
- Inference: ~20-50 FPS (faster CPU, no OS bloat)
- Evolution: ~5-10 candidates per minute (parallelized)

---

## 7. Troubleshooting

### Error: `Could not find openvino_tssn_extension.so`
- Make sure you ran the make command in Step 4.
- Check the path: !ls src/custom_ops/build/

### Error: `RuntimeError: Plugin not found`
- The extension isn't loading. Add debug prints:
  `python
  print(f"Extension Path: {EXTENSION_PATH}")
  print(f"File Exists: {EXTENSION_PATH.exists()}")
  `

### Error: `Out of Memory`
- Reduce the batch size or model size.
- Free up GPU memory: `torch.cuda.empty_cache()`

---

## 8. Next Steps After Migration

Once you have the pipeline running on Colab:
1. **Benchmark**: Compare CPU inference speed (Colab vs Local).
2. **Full Evolution Run**: Let the Metabolic War run for 100+ generations.
3. **Checkpoint**: Save intermediate models to Drive every 10 generations.
4. **Perplexity Testing**: Use `quick_accuracy_test.py` to verify the model still speaks English.

---

*For questions, see the main project wiki: `CYBERSPORE_WIKI.md`*
