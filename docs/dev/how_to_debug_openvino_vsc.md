OpenVINO Debugging Guide using VS Code
========================

This guide explains how to build and debug OpenVINO using Visual Studio Code,
including Python-to-C++ debugging, Python-only debugging, and native C++
runtime debugging workflows.

## 1. Prepare the workspace

Install required Linux packages:

```bash
sudo apt update && sudo apt install -y \
  git cmake ninja-build g++ python3-dev python3-venv \
  libssl-dev libzstd-dev pkg-config build-essential \
  binutils git-lfs lld-14

git lfs install
```

Create a clean Python environment:

```bash
python3 -m venv ~/ov_dbg_env
source ~/ov_dbg_env/bin/activate
pip install --upgrade pip
pip install numpy
```

## 2. Clone OpenVINO sources

```bash
# Create workspace
mkdir -p ~/code/ov_src_debug
cd ~/code/ov_src_debug

# Clone repo
git clone https://github.com/openvinotoolkit/openvino.git
cd openvino

# Init submodules
git submodule update --init --recursive

```

## 3. Configure Debug build

```bash
mkdir -p ~/code/ov_build_debug
cd ~/code/ov_build_debug
               
# Configure CMake build
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DENABLE_PYTHON=ON \
  -DPython3_EXECUTABLE=$(which python3) \
  ../ov_src_debug/openvino
```

For a faster build without GPU support:

```bash
-DENABLE_INTEL_GPU=OFF
```

## 4. Build the project

```bash
cd ~/code/ov_build_debug
ninja -j"$(nproc)"
```

## 5. Link OpenVINO Debug Build to Environment

Create `~/code/ov_build_debug/setup_debug_env.sh`:

```bash
# Link OpenVINO Debug build to environment
OV_DEBUG=~/code/ov_src_debug/openvino/bin/intel64/Debug
export PYTHONPATH="$OV_DEBUG/python:$PYTHONPATH"
export LD_LIBRARY_PATH="$OV_DEBUG:$LD_LIBRARY_PATH"
```

Make it executable:

```bash
chmod +x ~/code/ov_build_debug/setup_debug_env.sh
```

Add it to your bash profile:

```bash
echo "source ~/code/ov_build_debug/setup_debug_env.sh" >> ~/.bashrc
source ~/.bashrc
```

Test:

```bash
python3 -c "import openvino; print('OpenVINO Debug OK')"
```

## 6. Open the project in VS Code

1. Install extensions: C/C++, Python, WSL (for Windows)
2. Open VS Code.
3. For Windows + WSL:
   - Open Remote Explorer
   - Select WSL: Ubuntu
   - Ensure the status bar shows WSL: Ubuntu
4. Open folder: `~/code/ov_src_debug/openvino`

## 7. Configure Debugging

This section demonstrates how to debug OpenVINO using VS Code.

Supported debug modes:

- Python → C++ Debug (Launch Mode):
  Starts execution from a Python entry point and allows stepping into the
  native OpenVINO C++ backend.

- Python-Only Debug (Python Debugger):
  Executes and debugs Python-side logic exclusively, without entering
  native C++ layers.

- C++-Only Debug (Native Engine):
  Runs and debugs the OpenVINO Runtime C++ inference engine directly,
  without invoking Python bindings.

### Preparations

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
  ]
}
```

Create `debug_compile_model.py`:

```python
from openvino import Core
import openvino

print("=== OpenVINO Python → C++ debug demo ===")
print("OpenVINO loaded from:", openvino.__file__)

core = Core()
print("[Python] Core initialized successfully")

model_xml = "model.xml"
model_bin = "model.bin"
print(f"[Python] Reading model: {model_xml}, {model_bin}")
model = core.read_model(model_xml, model_bin)
print("[Python] Model read successfully")

print("[Python] Compiling model on CPU...")
compiled_model = core.compile_model(model, "CPU")
print("[Python] Model compiled successfully")

print("=== Debug demo finished ===")
```

### Python → C++ Debug (Launch Mode)

```json
{
  "name": "Launch OpenVINO Python (C++ Debug)",
  "type": "cppdbg",
  "request": "launch",
  "program": "/usr/bin/python3",
  "args": ["/home/user/code/ov_src_debug/openvino/debug_compile_model.py"],
  "cwd": "/home/user/code/ov_src_debug/openvino",
  "MIMode": "gdb"
}
```

Usage:

1. Set breakpoints in files such as:
   `src/inference/src/cpp/core.cpp` (e.g., in `Core::Core`)
2. Select: "Launch OpenVINO Python (C++ Debug)"
3. Press F5

When Python executes `core = Core()`, the debugger will stop at the C++
breakpoint.

### Python-Only Debug (Python Debugger)

```json
{
  "name": "Python Debugger: debug_compile_model.py",
  "type": "debugpy",
  "request": "launch",
  "program": "/home/user/code/ov_src_debug/openvino/debug_compile_model.py",
  "console": "integratedTerminal"
}
```

Usage:

1. Set breakpoints inside `debug_compile_model.py`
2. Select "Python Debugger: debug_compile_model.py"
3. Press F5
4. Inspect variables and call stack

### C++-Only Debug (Native Engine)

```json
{
  "name": "Debug OpenVINO Core (C++ Only)",
  "type": "cppdbg",
  "request": "launch",
  "program": "/home/user/code/ov_src_debug/openvino/bin/intel64/Debug/benchmark_app",
  "args": [
    "-m", "/home/user/code/ov_src_debug/openvino/model/test.xml",
    "-d", "CPU"
  ],
  "cwd": "/home/user/code/ov_src_debug/openvino/bin/intel64/Debug",
  "MIMode": "gdb"
}
```

Usage:

1. Set breakpoints inside OpenVINO Runtime sources
   (e.g., `src/inference/src/core.cpp`)
2. Select "Debug OpenVINO Core (C++ Only)"
3. Press F5
4. Debugger stops in native C++.

## See Also

- [OpenVINO™ README](../../README.md)
- [Developer documentation](./index.md)
- [OpenVINO debug capabilities](./debug_capabilities.md)
