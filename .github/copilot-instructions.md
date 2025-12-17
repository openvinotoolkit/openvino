# OpenVINO & Cyberspore Research - AI Coding Instructions

You are working in a hybrid environment containing the **OpenVINO source code** and a custom research project ("Cyberspore"/"TSSN") built on top of it.

## 1. Project Architecture & Context
- **OpenVINO Core**: The repository contains the full OpenVINO C++ source and Python bindings.
- **Research Layer**: Custom Python scripts (e.g., `apply_incision.py`, `metabolic_war.py`, `evolutionary_cycle.py`) implement experimental pruning ("incision"), healing, and evolutionary algorithms on OpenVINO models.
- **Key Concepts**: 
  - **Incision**: Targeted pruning of model weights (specifically FFN Down Projection layers).
  - **TSSN**: Ternary Sparse Synaptic Neuron layers.
  - **Metabolic War**: Competitive learning/evolutionary simulation.

## 2. Environment & Build (Windows/PowerShell)
- **Shell**: Always use **PowerShell** (`powershell.exe`).
- **Environment Setup**: You MUST run or replicate the logic of `setup_dev_env.ps1` before running Python scripts.
  - **PATH**: Must include `bin\intel64\Release` and `temp\Windows_AMD64\tbb\bin`.
  - **PYTHONPATH**: Must point to the locally built Python package (e.g., `bin\intel64\Release\python`).
  - **OPENVINO_LIB_PATHS**: Set this env var to the Release bin path to satisfy DLL checks.
- **Build Artifacts**: Located in `bin/intel64/Release`.

## 3. Python Scripting Guidelines
- **Boilerplate**: All standalone scripts MUST include the path setup boilerplate to find the local OpenVINO build if not running from a pre-configured shell. See `apply_incision.py` for the canonical example.
- **OpenVINO API**: 
  - Use `import openvino as ov`.
  - Use `from openvino.runtime import opset10 as ops` for graph construction.
- **Model Manipulation**:
  - Use `ov.Core().read_model()` to load IR files (`.xml`).
  - Use `ov.save_model()` to save modified models.
  - Pruning/Incision involves identifying `MatMul` operations and modifying their input `Constant` nodes (weights).

## 4. Developer Workflows
- **Running Experiments**:
  1. Ensure environment is set (`. .\setup_dev_env.ps1`).
  2. Run script: `python apply_incision.py`.
- **Debugging**:
  - Use VS Code launch configurations. Ensure `env` in `launch.json` mirrors `setup_dev_env.ps1`.
  - `inspect_ir.py` is useful for analyzing model structure before/after incision.
- **Testing OpenVINO Changes**:
  - If modifying OpenVINO source (e.g., `Type.t2` support), rebuild using CMake/VS, then run tests using the local python package.

## 5. Key Files
- `setup_dev_env.ps1`: Critical for environment configuration.
- `apply_incision.py`: Reference implementation for model pruning/weight manipulation.
- `metabolic_war.py`: Simulation logic for TSSN layers.
- `PLAN.md`: Current development objectives.
