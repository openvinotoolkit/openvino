## ✨ Feature: Disable Layer Fusion (via Config or Env Var)

This PR adds the ability to **disable CPU layer fusion in OpenVINO** via a **runtime config or environment variable** to enable **detailed profiling**, **layer-level debugging**, and more transparent runtime graphs.

---

### ✅ What's Added

- 🔑 New runtime config key: `DISABLE_LAYER_FUSION`
- 🌍 Environment variable support: `DISABLE_LAYER_FUSION`
- 🧪 Test script: `test_disable_fusion.py` to validate behavior
- 🧠 Integrated with CPU Plugin (via `Config::applyDebugCapsProperties`)
- 🔍 Prints runtime graph to inspect fused vs unfused ops

---

### ⚙️ How to Use

#### ✅ Option 1 – Runtime Config (Recommended)
```python
from openvino.runtime import Core

core = Core()
core.set_property("DISABLE_LAYER_FUSION", "YES")  # Must be set BEFORE compile_model()

model = core.read_model("path/to/model.xml")
compiled_model = core.compile_model(model, "CPU")
```

This results in a runtime graph with **separate ops**, e.g.:
```
ExecutionNode     /layer1/layer1.0/conv1/Conv/WithoutBiases  
ExecutionNode     /layer1/layer1.0/conv2/Conv/WithoutBiases  
ExecutionNode     /ReLU_123  
...
```

---

#### ⚠️ Option 2 – Environment Variable (Currently Unreliable)

```cmd
set DISABLE_LAYER_FUSION=YES
```

❌ This **does not reliably disable fusion** in current testing (Windows, local builds).  
It appears the plugin reads environment variables **after plugin initialization**, which is too late.

---

### 🧪 Validation Result

Model used: `resnet-50-pytorch` (from OpenVINO Model Zoo)  
Test script: [`test_disable_fusion.py`](tests/test_disable_fusion.py)

| Method                    | Fusion Disabled? | Notes                                   |
|---------------------------|------------------|------------------------------------------|
| `core.set_property(...)`  | ✅ Yes            | Reliable, activates before compilation   |
| `set DISABLE_LAYER_FUSION=YES` | ❌ No         | Plugin likely reads it too late          |

> 🔍 When fusion is disabled, runtime graph includes individual `Conv`, `ReLU`, and `Pooling` ops, allowing precise inspection of layer-level execution.

---

### 📁 Note: No CSV Report is Generated

The `test_disable_fusion.py` script:
- ✅ Prints the runtime graph to the console  
- ❌ Does **not** generate performance reports or `.csv` files like `benchmark_detailed_counters_report.csv`

To generate such reports, use the official `benchmark_app`:
```bash
benchmark_app -m model.xml -d CPU -report_type detailed_counters -report_folder ./results
```

---

### 🔍 Technical Notes

- The `DISABLE_LAYER_FUSION` config is handled in `Config::applyDebugCapsProperties()`.
- The environment variable path **may work in future** if plugin initialization is refactored to read it earlier.
- This feature is especially helpful for:
  - Fine-grained profiling
  - Debugging fused patterns that hide internal ops
  - Teaching/demo purposes where full graph visibility is required

---

### 🚧 Future Work

- [ ] Investigate making `DISABLE_LAYER_FUSION` via env var effective at plugin load time
- [ ] Ensure consistent behavior across platforms (Windows/Linux)
- [ ] Optional: Add CLI flags or diagnostic tools for fusion toggling