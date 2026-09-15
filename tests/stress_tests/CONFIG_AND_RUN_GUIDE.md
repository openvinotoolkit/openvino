# OpenVINO Stress Tests - Configuration & Execution Guide (BKM)

This document describes the structure and format of `test_config.xml` used by the OpenVINO stress test suite (`StressUnitTests`, `StressMemLeaksTests`, `MemCheckTests`), along with common execution options, filtering recipes, and heterogeneous multi-model testing scenarios.

---

## 1. `test_config.xml` Format & Schema

The configuration file defines the test parameters matrix. The test harness generates the Cartesian product across all configured dimensions to instantiate parameterized test cases.

```xml
<attributes>
    <!-- Number of concurrent processes spawned via fork()/exec() -->
    <processes>
        <value>1</value>
        <value>2</value>
    </processes>

    <!-- Number of worker threads per process -->
    <threads>
        <value>1</value>
        <value>4</value>
    </threads>

    <!-- Number of scenario iterations/loops -->
    <iterations>
        <value>50</value>
    </iterations>

    <!-- Target OpenVINO devices/plugins -->
    <devices>
        <value>CPU</value>
        <value>NPU</value>
        <value>GPU</value>
    </devices>

    <!-- Optional global compilation configuration file (defaults to PERFORMANCE_HINT LATENCY if omitted) -->
    <compilation_config_file>/path/to/global_compilation_config.txt</compilation_config_file>

    <!-- Optional failure log collection settings -->
    <failure_logs_dir>/path/to/failure_logs</failure_logs_dir>
    <fw_log_path>/sys/kernel/debug/accel/0000:00:0b.0/fw_log</fw_log_path>

    <!-- Model IR definitions -->
    <models>
        <model name="heavy_model" path="heavy_model" full_path="/path/to/heavy_model.xml" precision="FP32" compilation_config="/path/to/heavy_compilation_config.txt" />
        <model name="light_model" path="light_model" full_path="/path/to/light_model.xml" precision="FP32" />
    </models>
</attributes>
```

### Element Breakdown

| XML Element | Sub-element / Attributes | Description |
|---|---|---|
| `<processes>` | `<value>N</value>` | Number of OS child processes to spawn concurrently. Default is `1`. |
| `<threads>` | `<value>N</value>` | Number of worker threads per process. Default is `1`. |
| `<iterations>` | `<value>N</value>` | Number of iterations/loops each worker runs. Default is `1`. |
| `<devices>` | `<value>DEV</value>` | OpenVINO plugin device: `CPU`, `GPU`, `NPU`, `AUTO`, `HETERO`, etc. |
| `<compilation_config_file>` | *Path string* | (*Optional*) Path to compilation configuration file applied globally across models. |
| `<models>` | `<model ... />` | List of OpenVINO IR models to test. |

#### `<model>` Attributes:
- **`full_path`** (*Required*): Absolute or relative filesystem path to the OpenVINO IR `.xml` file (the companion `.bin` file must reside in the same directory).
- **`path`** (*Required*): Short identifier used for GoogleTest test case naming.
- **`name`** (*Optional*): Model name, used during automated Open Model Zoo (OMZ) acquisition.
- **`precision`** (*Optional*): Precision descriptor (e.g. `FP32`, `FP16`, `INT8`).
- **`compilation_config`** (*Optional*): Path to a model-specific compilation config file. Overrides the global `<compilation_config_file>`.

### Compilation Configuration File Format

The compilation configuration file defines properties passed to `ov::Core::compile_model()`. If no configuration file is provided, tests default to `PERFORMANCE_HINT LATENCY`.

Both **JSON** (`.json`) and **plain text** (`.txt`) configuration file formats are supported.

#### A. JSON Format (Recommended for Multi-Device Configs)

Supports device-scoped blocks (`"NPU"`, `"GPU"`, `"CPU"`) as well as global top-level properties:

```json
{
    "PERFORMANCE_HINT": "LATENCY",
    "NPU": {
        "NPU_USE_NPUW": "YES",
        "NPUW_DEVICES": "NPU",
        "NPUW_FUNCALL_FOR_ALL": "YES",
        "NPU_COMPILER_DYNAMIC_QUANTIZATION": "YES",
        "NPU_QDQ_OPTIMIZATION": "NO",
        "NPUW_ENSURE_COMPATIBILITY": "YES",
        "NPU_ENABLE_STRIDES_FOR": "past_key_values.0.key,past_key_values.1.key"
    },
    "GPU": {
        "INFERENCE_PRECISION_HINT": "f16"
    },
    "CPU": {
        "NUM_STREAMS": "1",
        "INFERENCE_NUM_THREADS": "4"
    }
}
```

#### B. Plain Text Format (`compilation_config.txt`)

```text
# Global properties (applied to all devices)
PERFORMANCE_HINT LATENCY
NUM_STREAMS 2
INFERENCE_NUM_THREADS 4

# Or device-scoped properties:
# NPU PERFORMANCE_HINT LATENCY
# CPU NUM_STREAMS 4
```

Supported syntax:
- JSON with device-scoped objects (`"NPU": { ... }`) or flat JSON object
- Plain text `KEY VALUE` or `KEY=VALUE`
- Plain text `DEVICE KEY VALUE`
- Comments starting with `#` or `//`
- Quoted string values (e.g. `PERFORMANCE_HINT "LATENCY"`)

---

## 2. Test Scenarios Reference

`StressUnitTests` provides both single-model lifecycle stress scenarios and multi-model heterogeneous concurrency scenarios:

| Scenario | Target Test Suite | Description |
|---|---|---|
| `stress_load_unload` | `UnitTestSuite` | Repeated model compilation, inference, and destruction. |
| `stress_parallel_infer` | `UnitTestSuite` | Parallel threads sharing a single `ov::CompiledModel`. |
| `stress_concurrent_load_infer` | `UnitTestSuite` | Active inference concurrent with background model compilation. |
| `stress_import_export` | `UnitTestSuite` | Model export to blob, import from blob, and inference. |
| `stress_mid_flight_cancel` | `UnitTestSuite` | Cancels active inference while other worker requests run. |
| `stress_memory_pressure` | `UnitTestSuite` | Compiles multiple concurrent model instances to stress device memory. |
| `stress_destroy_compiled_model` | `UnitTestSuite` | Destructs `ov::CompiledModel` while an inference request is executing. |
| `stress_multiple_cores` | `UnitTestSuite` | Rapid simultaneous creation and destruction of `ov::Core` objects. |
| `stress_heterogeneous_concurrent_infer` | `UnitTestSuiteMultiModel` | Concurrently runs heavy and lightweight models across parallel threads. |
| `stress_heterogeneous_concurrent_processes` | `UnitTestSuiteMultiModel` | Concurrently runs different models in distinct OS child processes. |

---

## 3. Sample Execution Commands

Set up library paths:
```bash
export REBASE_DIR=/home/celadon/workspace/rebase_26ww36.1.1
export BIN_DIR=$REBASE_DIR/openvino/tests/stress_tests/bin/intel64/Release
export LD_LIBRARY_PATH=$REBASE_DIR/openvino/bin/intel64/Release:$REBASE_DIR/one-tbb/one-tbb-install-linux/lib:$LD_LIBRARY_PATH
```

### A. List All Generated Tests
```bash
$BIN_DIR/StressUnitTests \
    --test_conf=/path/to/test_config.xml \
    --gtest_list_tests
```

### B. Recommended Execution Pattern (Combined XML + Console + Failure Logs)
To ensure complete test accounting, CI integration, and debugging visibility, run with `--gtest_output=xml:...` while piping output through `tee`:

```bash
# Using direct command line:
$BIN_DIR/StressUnitTests \
    --test_conf=/path/to/test_config.xml \
    --gtest_filter='StressUnitTests/UnitTestSuite.stress_load_unload/*' \
    --gtest_output=xml:./test_output/test_results.xml \
    --failure_logs_dir=./test_output/failure_logs 2>&1 | tee ./test_output/test_run.log

# Or using the automated runner script:
./scripts/run_stress_unittests.sh \
    --test_conf /path/to/test_config.xml \
    --gtest_filter 'StressUnitTests/UnitTestSuite.stress_load_unload/*' \
    --output_dir ./test_output
```

### C. Run All Coordinated Single-Model Stress Tests
```bash
# Run with default LATENCY performance hint
$BIN_DIR/StressUnitTests \
    --test_conf=/path/to/test_config.xml \
    --gtest_filter='StressUnitTests/UnitTestSuite.stress_*'

# Run with custom compilation configuration file override
$BIN_DIR/StressUnitTests \
    --test_conf=/path/to/test_config.xml \
    --compilation_config_file=/path/to/compilation_config.txt \
    --gtest_filter='StressUnitTests/UnitTestSuite.stress_*'
```

### D. Run Specific Single-Model Scenarios on NPU / CPU / GPU
```bash
# Parallel inference on NPU with compilation config
$BIN_DIR/StressUnitTests \
    --test_conf=/path/to/test_config.xml \
    --compilation_config_file=/path/to/npu_compilation_config.txt \
    --gtest_filter='StressUnitTests/UnitTestSuite.stress_parallel_infer/*NPU*'

# Concurrent compile & infer on CPU
$BIN_DIR/StressUnitTests \
    --test_conf=/path/to/test_config.xml \
    --gtest_filter='StressUnitTests/UnitTestSuite.stress_concurrent_load_infer/*CPU*'
```

### E. Run Multi-Model Heterogeneous Stress Scenarios (Approach B)

#### 1. Thread-Level Concurrency (Heavy + Light Models in Parallel Threads):
```bash
$BIN_DIR/StressUnitTests \
    --test_conf=/path/to/test_config.xml \
    --gtest_filter='StressUnitTests/UnitTestSuiteMultiModel.stress_heterogeneous_concurrent_infer/*'
```

#### 2. Process-Level Concurrency (Heavy + Light Models across Separate Processes):
```bash
$BIN_DIR/StressUnitTests \
    --test_conf=/path/to/test_config.xml \
    --gtest_filter='StressUnitTests/UnitTestSuiteMultiModel.stress_heterogeneous_concurrent_processes/*'
```

### F. Standalone Child Execution (Without GoogleTest Harness)
Direct execution without XML parsing:

```bash
# Single model standalone run (default LATENCY hint)
$BIN_DIR/StressUnitTests \
    --stress_child \
    --stress_scenario=stress_parallel_infer \
    --stress_model=/path/to/model.xml \
    --stress_device=NPU \
    --stress_iterations=200 \
    --stress_threads=4

# Single model standalone run with custom compilation config
$BIN_DIR/StressUnitTests \
    --stress_child \
    --stress_scenario=stress_parallel_infer \
    --stress_model=/path/to/model.xml \
    --stress_device=NPU \
    --stress_iterations=200 \
    --stress_threads=4 \
    --stress_compilation_config=/path/to/compilation_config.txt

# Multi-model concurrent standalone run
$BIN_DIR/StressUnitTests \
    --stress_child \
    --stress_scenario=stress_heterogeneous_concurrent_infer \
    --stress_model=/path/to/heavy_model.xml \
    --stress_model2=/path/to/light_model.xml \
    --stress_device=NPU \
    --stress_iterations=200 \
    --stress_threads=4 \
    --stress_compilation_config=/path/to/heavy_config.txt \
    --stress_compilation_config2=/path/to/light_config.txt
```

---

## 4. Real-Time Hardware Metrics Monitoring & Execution Summary Reports

`StressUnitTests` and `StressMemLeaksTests` include a dedicated background monitoring thread (`MetricsMonitor`) that runs concurrently during test execution to sample CPU and NPU hardware utilization, calculate active stress times, and produce summary reports.

### Features:
- **Live Status Heartbeat:** Streams live progress to the console every 2s (`[ METRICS ] Active: <test> | Elapsed: 4.2s | CPU: 12.3% | NPU: 78.4% | NPU Stressed: 3.8s`).
- **NPU Stress Time & Duty Cycle:** Accurately calculates total active NPU hardware execution time (in seconds) and duty cycle percentage ($\frac{\text{NPU Stress Time}}{\text{Test Run Time}} \times 100\%$) via Intel VPU/accel telemetry counters (`npu_busy_time_us`).
- **Summary Reports:** Automatically writes formatted ASCII summary reports (`stress_test_metrics_report.txt`) and structured machine-readable JSON reports (`stress_test_metrics_report.json`).

### Report Format Example:
```text
========================================================================================================
                           OpenVINO Stress Tests - Metrics & Execution Summary Report                   
========================================================================================================
Generated:          2026-09-15 10:04:00 UTC
Report Directory:   ./test_results
NPU Telemetry:      /sys/bus/pci/drivers/intel_vpu/0000:00:0b.0/npu_busy_time_us
Total Tests:        8 (8 Passed, 0 Failed)
Total Run Time:     245.30 s
Total NPU Stress:   210.40 s (85.8% Active NPU Duty Cycle)
========================================================================================================

TEST CASE                                                   STATUS    RUN TIME (s)  CPU (AVG/PEAK)   NPU (AVG/PEAK)   NPU STRESS (s)  NPU DUTY %
-------------------------------------------------------------------------------------------------------------------------------------------------
StressUnitTests/UnitTestSuite.stress_parallel_infer/NP1...   PASSED    30.50         14.2/28.0%       85.4/98.2%       26.10           85.6%
StressUnitTests/UnitTestSuite.stress_load_unload/NP1...      PASSED    310.18        88.5/99.0%       0.2/1.5%         0.62            0.2%
-------------------------------------------------------------------------------------------------------------------------------------------------
```

### Configuration & CLI Options:
- **`--enable_metrics=true|false`**: Enable/disable background metrics monitoring (default: `true`).
- **`--metrics_interval_ms=500`**: Sampling frequency in milliseconds (default: `500`).
- **`--metrics_report_dir=/path/to/dir`**: Directory where `stress_test_metrics_report.txt` and `.json` are written (default: `./test_results`).
- **`--metrics_live_updates=true|false`**: Enable/disable real-time console progress heartbeats (default: `true`).

---

## 5. Automated Failure Diagnostics & Log Collection

`StressUnitTests` and `StressMemLeaksTests` automatically monitor each test in the background and capture kernel `dmesg` logs, NPU firmware logs (`/sys/kernel/debug/accel/*/fw_log`), and test failure reports whenever any test fails.

### Output Files on Failure:
When a test fails, the harness automatically writes the following files to `./test_failure_logs/` (or the configured directory):
- `<test_name>_report.txt`: Test duration, start/end timestamps, and GTest assertion / crash failure details.
- `<test_name>_dmesg.log`: Kernel ring buffer message snapshot for kernel driver inspection.
- `<test_name>_fw_log.log`: NPU firmware debugfs log delta generated during the test execution window.

### Configuration & CLI Options:
- **`--collect_failure_logs=true|false`**: Enable/disable automatic failure log collection (default: `true`).
- **`--failure_logs_dir=/path/to/dir`**: Custom destination directory for logs (default: `./test_failure_logs`).
- **`--fw_log_path=/path/to/fw_log`**: Explicit path to NPU firmware log (defaults to auto-detecting `/sys/kernel/debug/accel/*/fw_log`).

---

## 6. Workload Execution & Hardware Utilization Guide (NPU vs Host CPU)

When executing tests on target accelerators such as the **NPU**, understanding where each test phase executes is critical for diagnosing hardware utilization and performance:

### A. Host CPU Model Compilation vs NPU Hardware Execution
- **`create_compiled_model` (`Create ExecutableNetwork`)**:
  - The test invokes `ov::Core::compile_model(..., "NPU")`.
  - The OpenVINO NPU compiler plugin (`libopenvino_intel_npu_compiler.so`) executes **entirely on the host CPU** to parse IR layers, perform graph optimization/quantization transformations, and produce the compiled binary blob.
  - **Expected Hardware Activity:** NPU execution engines remain idle (~0% utilization) while host CPU utilization increases.
- **`infer_request_inference` & `stress_*` Scenarios**:
  - Dispatches inference execution requests to the NPU driver and VPU hardware engine via the Level-Zero / UMD driver (`libze_intel_npu.so`).
  - **Expected Hardware Activity:** High NPU engine/memory utilization visible in monitoring tools.

### B. Test Suite Workload Breakdown

| Test Name / Scenario | Host CPU Compilation | Buffer Allocation | NPU Hardware Inference | Notes |
|---|:---:|:---:|:---:|---|
| `load_unload_plugin` | ❌ | ❌ | ❌ | Tests plugin library load & unload lifecycle. |
| `read_network` | ❌ | ❌ | ❌ | Tests IR XML/BIN parsing and frontend reading. |
| `cnnnetwork_reshape_*` | ❌ | ❌ | ❌ | Tests dynamic/static shape propagation. |
| `create_compiled_model` | ✔️ (Host CPU) | ❌ | ❌ | Compiles model to blob on CPU; no NPU infer. |
| `create_infer_request` | ✔️ (Host CPU) | ✔️ | ❌ | Compiles and allocates IO memory buffers only. |
| `infer_request_inference` | ✔️ (Host CPU) | ✔️ | ✔️ (NPU) | Executes synchronous/asynchronous inference on NPU. |
| `stress_parallel_infer` | ✔️ (1x Init) | ✔️ | ✔️ (NPU) | Parallel worker threads dispatching NPU inferences. |
| `stress_load_unload` | ✔️ (Repeated) | ✔️ | ✔️ (NPU) | Continuous compile, infer on NPU, and teardown. |
| `stress_concurrent_load_infer` | ✔️ (Background) | ✔️ | ✔️ (NPU) | Background thread compiles while workers infer on NPU. |
| `stress_heterogeneous_*` | ✔️ | ✔️ | ✔️ (NPU) | Multi-model concurrent inference across threads/procs. |

### C. Monitoring NPU Utilization in Real Time

To verify active NPU hardware execution during inference stress runs:

1. **Launch Real-Time NPU Hardware Monitor (in a separate terminal):**
   ```bash
   # Using intel-npu-smi tool:
   watch -n 0.5 /path/to/release/intel-npu-smi

   # Or using npu-utilization script:
   /path/to/release/npu-utilization.sh
   ```

2. **Execute Inference Workload on NPU:**
   ```bash
   export LD_LIBRARY_PATH=/path/to/release:$LD_LIBRARY_PATH

   # Run multi-threaded parallel inference
   StressUnitTests --test_conf=/path/to/test_config.xml \
       --gtest_filter='StressUnitTests/UnitTestSuite.stress_parallel_infer/*NPU*'

   # Run multi-model concurrent inference
   StressUnitTests --test_conf=/path/to/test_config.xml \
       --gtest_filter='StressUnitTests/UnitTestSuiteMultiModel.stress_heterogeneous_concurrent_infer/*'
   ```

### D. Understanding Progress Logs
- **`"[ INFO ] Half of the test have already passed"`**: Emitted at iteration $N / 2$ of each test loop (e.g., at iteration 25 for `<iterations><value>50</value></iterations>`) to signal test liveness and 50% completion checkpoint.


