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

Sample configuration file format (`compilation_config.txt`):
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
- `KEY VALUE` or `KEY=VALUE`
- `DEVICE KEY VALUE`
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

### B. Run All Coordinated Single-Model Stress Tests
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

### C. Run Specific Single-Model Scenarios on NPU / CPU / GPU
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

### D. Run Multi-Model Heterogeneous Stress Scenarios (Approach B)

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

### E. Standalone Child Execution (Without GoogleTest Harness)
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
