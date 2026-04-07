# Unit Tests for `ov::intel_cpu::multi_app_thread_sync_execution` Property

## Background

The `feature/multithread_sync_property` branch introduces a new CPU-specific property:

```
ov::intel_cpu::multi_app_thread_sync_execution  ("CPU_MULTI_APP_THREAD_SYNC_EXECUTION")
Type:    bool
Default: false
Mutability: RW (plugin level), RO (compiled model level)
```

### What it does

When `true`, `AsyncInferRequest` bypasses the default `IAsyncInferRequest::infer()` dispatch and instead routes inference directly through `m_stream_executor->execute()`. This is intended for multi-application, thread-synchronised scenarios where the caller manages thread scheduling externally (e.g., `multithread_sync_app`).

### Files changed by the branch

| File | Change |
|------|--------|
| `include/openvino/runtime/intel_cpu/properties.hpp` | Declares `multi_app_thread_sync_execution` property constant |
| `src/plugins/intel_cpu/src/config.h` | Adds `bool multiAppThreadSyncExecution = false` to `Config` struct |
| `src/plugins/intel_cpu/src/config.cpp` | `readProperties()` parses the key; `updateProperties()` writes `"YES"`/`"NO"` to `_config` |
| `src/plugins/intel_cpu/src/compiled_model.cpp` | Exposes property via `get_property()` and lists it in `supported_properties` |
| `src/plugins/intel_cpu/src/plugin.cpp` | Exposes property via plugin-level `get_ro_property()` |
| `src/plugins/intel_cpu/src/async_infer_request.cpp` | New constructor param `multi_app_thread_sync_execution`; switches execution path |

---

## Test Plan

### Framework Selection

Two test targets cover different layers of the stack:

| Target binary | Location | Purpose |
|---|---|---|
| `ov_cpu_unit_tests` | `src/plugins/intel_cpu/tests/unit/` | Config struct parsing, no OV runtime — fast and isolated |
| `ov_cpu_func_tests` | `src/plugins/intel_cpu/tests/functional/custom/behavior/` | Full OV runtime: `ov::Core`, `ov::CompiledModel`, inference |

---

### Group A — Unit Tests (6 tests)

**New file**: `src/plugins/intel_cpu/tests/unit/multi_app_thread_sync_test.cpp`

Tests exercise `ov::intel_cpu::Config::readProperties()` and `Config::updateProperties()` directly without spinning up the OV plugin runtime. This follows the same pattern as `tests/unit/streams_info/streams_e2e_test.cpp`.

| # | Test name | One-liner |
|---|---|---|
| A1 | `Config_DefaultMultiAppThreadSyncIsFalse` | Construct a default `Config` object; assert `multiAppThreadSyncExecution == false` |
| A2 | `Config_ReadProperties_SetTrue` | Call `readProperties({{"CPU_MULTI_APP_THREAD_SYNC_EXECUTION", true}})`; assert field is `true` |
| A3 | `Config_ReadProperties_SetFalse` | Set `true` then call again with `false`; assert field toggles back to `false` |
| A4 | `Config_ReadProperties_InvalidValueThrows` | Pass non-bool string `"maybe"`; assert `ov::Exception` is thrown |
| A5 | `Config_UpdateProperties_TrueWritesYES` | Set field to `true`, call `updateProperties()`; assert `_config["CPU_MULTI_APP_THREAD_SYNC_EXECUTION"] == "YES"` |
| A6 | `Config_UpdateProperties_FalseWritesNO` | Default (false), call `updateProperties()`; assert `_config["CPU_MULTI_APP_THREAD_SYNC_EXECUTION"] == "NO"` |

**Implementation approach**: Direct instantiation of `ov::intel_cpu::Config`, no mocking required. New file is auto-discovered by `ov_add_test_target(ROOT ...)` glob in `unit/CMakeLists.txt` — no CMakeLists change needed.

---

### Group B — Functional Tests (9 new tests + 2 existing test modifications)

#### B1 — Plugin-level tests: `custom/behavior/ov_plugin/properties.cpp`

| # | Test name | One-liner |
|---|---|---|
| B1-mod | **Modify** `smoke_PluginAllSupportedPropertiesAreAvailable` | Add `RW_property(ov::intel_cpu::multi_app_thread_sync_execution.name())` to `expectedSupportedProperties` |
| B2 | `smoke_CpuPluginMultiAppThreadSyncDefaultIsFalse` | `core.get_property("CPU", ov::intel_cpu::multi_app_thread_sync_execution)` returns `false` by default |
| B3 | `smoke_CpuPluginSetMultiAppThreadSyncTrue` | `core.set_property` with `true`; `get_property` round-trips to `true` |
| B4 | `smoke_CpuPluginSetMultiAppThreadSyncFalse` | Same round-trip with `false`; verifies property is writable in both directions |

#### B2 — Compiled model tests: `custom/behavior/ov_executable_network/properties.cpp`

| # | Test name | One-liner |
|---|---|---|
| B5-mod | **Modify** `smoke_CpuExecNetworkSupportedPropertiesAreAvailable` | Add `RO_property(ov::intel_cpu::multi_app_thread_sync_execution.name())` to `expectedSupportedProperties` |
| B6 | `smoke_CpuExecNetworkMultiAppThreadSyncDefaultIsFalse` | Compile without setting property; `compiled_model.get_property(...)` returns `false` |
| B7 | `smoke_CpuExecNetworkMultiAppThreadSyncSetTrue` | Compile with `{multi_app_thread_sync_execution, true}` in config; `get_property` reads back `true` |
| B8 | `smoke_CpuExecNetworkMultiAppThreadSyncSetFalse` | Compile with explicit `false`; `get_property` reads back `false` |
| B9 | `smoke_CpuExecNetworkMultiAppThreadSyncInferNoThrow` | Compile with `true`, create infer request, bind input tensor, call `infer()` — no throw, non-empty output |
| B10 | `smoke_CpuExecNetworkMultiAppThreadSyncResultsMatchDefault` | Run same model with `true` and `false`; outputs are element-wise identical (only dispatch path differs) |
| B11 | `smoke_CpuExecNetworkMultiAppThreadSyncWithStreams` | Set `num_streams=4` + run inference with both `true` and `false`; assert no-throw, non-empty output, and element-wise equal outputs |

---

### Summary

| Framework | File | New tests | Existing test modifications |
|---|---|---|---|
| `ov_cpu_unit_tests` | new `multi_app_thread_sync_test.cpp` | 6 | 0 |
| `ov_cpu_func_tests` | `ov_plugin/properties.cpp` | 3 | 1 |
| `ov_cpu_func_tests` | `ov_executable_network/properties.cpp` | 6 | 1 |
| **Total** | | **15** | **2** |

---

### How to run

```bash
# Build (from build dir):
cmake --build . --parallel --target ov_cpu_unit_tests ov_cpu_func_tests

# Unit tests only (fast, no OV runtime model):
./bin/intel64/Release/ov_cpu_unit_tests --gtest_filter="*MultiAppThread*"

# Functional property tests (plugin level):
./bin/intel64/Release/ov_cpu_func_tests --gtest_filter="*MultiAppThreadSync*"

# Or via CTest:
ctest -R "ov_cpu_unit_tests|ov_cpu_func_tests" -V --test-dir build/ \
      --gtest_filter="*MultiAppThread*"
```

---

## Implementation

### Patch file

Patch located at: `/workspace/sw/jvincent/WORK/MULSTRPR/multi_app_thread_sync_tests.patch`

Generated with:
```bash
cd /workspace/sw/jvincent/WORK/MULSTRPR/openvino
git diff HEAD > ../multi_app_thread_sync_tests.patch
```

Applied with:
```bash
git apply ../multi_app_thread_sync_tests.patch
```

---

## Build Notes

### Pre-existing build infrastructure gap fixed

The unit test CMakeLists was missing the snippets test include path, causing compile failures
for unrelated `snippets_transformations/` tests. Fixed by adding one line to
`src/plugins/intel_cpu/tests/unit/CMakeLists.txt`:

```cmake
# After fix (PRIVATE includes section):
$<TARGET_PROPERTY:openvino::snippets,SOURCE_DIR>/include
$<TARGET_PROPERTY:openvino::snippets,SOURCE_DIR>/tests/include   # <-- added
```

`snippets_test_utils` (built under `src/common/snippets/tests/`) provides headers like
`lir_test_utils.hpp` and `lowering_utils.hpp` that the CPU snippets transformation tests include
directly. This path was missing from the include list.

### CMake configuration used

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_INTEL_CPU=ON \
  -DENABLE_INTEL_GPU=OFF \
  -DENABLE_INTEL_NPU=OFF \
  -DENABLE_TESTS=ON \
  -DENABLE_FUNCTIONAL_TESTS=ON \
  -DENABLE_HETERO=OFF \
  -DENABLE_AUTO=OFF \
  -DENABLE_AUTO_BATCH=OFF \
  -DENABLE_MULTI=OFF \
  -DENABLE_TEMPLATE=ON \
  -DENABLE_SAMPLES=ON \
  -DENABLE_PYTHON=OFF
```

> **Note:** `ENABLE_FUNCTIONAL_TESTS=ON` is required because `snippets_test_utils` is only built
> when functional tests are enabled (it is a dependency of `ov_cpu_unit_tests`).
> `ENABLE_HETERO/AUTO/MULTI=OFF` avoids the `openvino::funcSharedTests` linker error from those
> plugin functional tests.

### Build commands

```bash
cmake --build . --parallel --target ov_cpu_unit_tests
cmake --build . --parallel --target ov_cpu_func_tests
```

Both targets linked successfully with no errors.

---

## Test Results

### Unit tests — `ov_cpu_unit_tests`  ✅ 6/6 PASSED

```
./bin/intel64/Release/ov_cpu_unit_tests --gtest_filter="*MultiAppThread*"

[==========] Running 6 tests from 1 test suite.
[----------] 6 tests from MultiAppThreadSyncTest
[ RUN      ] MultiAppThreadSyncTest.Config_DefaultMultiAppThreadSyncIsFalse
[       OK ] MultiAppThreadSyncTest.Config_DefaultMultiAppThreadSyncIsFalse (10 ms)
[ RUN      ] MultiAppThreadSyncTest.Config_ReadProperties_SetTrue
[       OK ] MultiAppThreadSyncTest.Config_ReadProperties_SetTrue (4 ms)
[ RUN      ] MultiAppThreadSyncTest.Config_ReadProperties_SetFalse
[       OK ] MultiAppThreadSyncTest.Config_ReadProperties_SetFalse (0 ms)
[ RUN      ] MultiAppThreadSyncTest.Config_ReadProperties_InvalidValueThrows
[       OK ] MultiAppThreadSyncTest.Config_ReadProperties_InvalidValueThrows (21 ms)
[ RUN      ] MultiAppThreadSyncTest.Config_UpdateProperties_TrueWritesYES
[       OK ] MultiAppThreadSyncTest.Config_UpdateProperties_TrueWritesYES (0 ms)
[ RUN      ] MultiAppThreadSyncTest.Config_UpdateProperties_FalseWritesNO
[       OK ] MultiAppThreadSyncTest.Config_UpdateProperties_FalseWritesNO (0 ms)
[----------] 6 tests from MultiAppThreadSyncTest (35 ms total)
[==========] 6 tests ran. (35 ms total)
[  PASSED  ] 6 tests.
```

### Functional tests — `ov_cpu_func_tests`  ✅ 9/9 new + 2/2 modified PASSED

```
./bin/intel64/Release/ov_cpu_func_tests --gtest_filter="*MultiAppThreadSync*"

[==========] Running 9 tests from 1 test suite.
[----------] 9 tests from OVClassConfigTestCPU
[ RUN      ] OVClassConfigTestCPU.smoke_CpuExecNetworkMultiAppThreadSyncDefaultIsFalse
[       OK ] OVClassConfigTestCPU.smoke_CpuExecNetworkMultiAppThreadSyncDefaultIsFalse (434 ms)
[ RUN      ] OVClassConfigTestCPU.smoke_CpuExecNetworkMultiAppThreadSyncSetTrue
[       OK ] OVClassConfigTestCPU.smoke_CpuExecNetworkMultiAppThreadSyncSetTrue (23 ms)
[ RUN      ] OVClassConfigTestCPU.smoke_CpuExecNetworkMultiAppThreadSyncSetFalse
[       OK ] OVClassConfigTestCPU.smoke_CpuExecNetworkMultiAppThreadSyncSetFalse (22 ms)
[ RUN      ] OVClassConfigTestCPU.smoke_CpuExecNetworkMultiAppThreadSyncInferNoThrow
[       OK ] OVClassConfigTestCPU.smoke_CpuExecNetworkMultiAppThreadSyncInferNoThrow (26 ms)
[ RUN      ] OVClassConfigTestCPU.smoke_CpuExecNetworkMultiAppThreadSyncResultsMatchDefault
[       OK ] OVClassConfigTestCPU.smoke_CpuExecNetworkMultiAppThreadSyncResultsMatchDefault (44 ms)
[ RUN      ] OVClassConfigTestCPU.smoke_CpuExecNetworkMultiAppThreadSyncWithStreams
[       OK ] OVClassConfigTestCPU.smoke_CpuExecNetworkMultiAppThreadSyncWithStreams (25 ms)
[ RUN      ] OVClassConfigTestCPU.smoke_CpuPluginMultiAppThreadSyncDefaultIsFalse
[       OK ] OVClassConfigTestCPU.smoke_CpuPluginMultiAppThreadSyncDefaultIsFalse (1 ms)
[ RUN      ] OVClassConfigTestCPU.smoke_CpuPluginSetMultiAppThreadSyncTrue
[       OK ] OVClassConfigTestCPU.smoke_CpuPluginSetMultiAppThreadSyncTrue (2 ms)
[ RUN      ] OVClassConfigTestCPU.smoke_CpuPluginSetMultiAppThreadSyncFalse
[       OK ] OVClassConfigTestCPU.smoke_CpuPluginSetMultiAppThreadSyncFalse (1 ms)
[----------] 9 tests from OVClassConfigTestCPU (580 ms total)
[==========] 9 tests ran. (585 ms total)
[  PASSED  ] 9 tests.
```

```
./bin/intel64/Release/ov_cpu_func_tests \
  --gtest_filter="*smoke_PluginAllSupportedPropertiesAreAvailable*:*smoke_CpuExecNetworkSupportedPropertiesAreAvailable*"

[==========] Running 2 tests from 1 test suite.
[ RUN      ] OVClassConfigTestCPU.smoke_CpuExecNetworkSupportedPropertiesAreAvailable
[       OK ] OVClassConfigTestCPU.smoke_CpuExecNetworkSupportedPropertiesAreAvailable (37 ms)
[ RUN      ] OVClassConfigTestCPU.smoke_PluginAllSupportedPropertiesAreAvailable
[       OK ] OVClassConfigTestCPU.smoke_PluginAllSupportedPropertiesAreAvailable (2 ms)
[==========] 2 tests ran. (43 ms total)
[  PASSED  ] 2 tests.
```

---

## Final Summary

| Category | Binary | Tests | Result |
|---|---|---|---|
| Unit — Config default | `ov_cpu_unit_tests` | A1 | ✅ PASSED |
| Unit — readProperties true | `ov_cpu_unit_tests` | A2 | ✅ PASSED |
| Unit — readProperties toggle | `ov_cpu_unit_tests` | A3 | ✅ PASSED |
| Unit — invalid value throws | `ov_cpu_unit_tests` | A4 | ✅ PASSED |
| Unit — updateProperties YES | `ov_cpu_unit_tests` | A5 | ✅ PASSED |
| Unit — updateProperties NO | `ov_cpu_unit_tests` | A6 | ✅ PASSED |
| Func — plugin supported list | `ov_cpu_func_tests` | B1-mod | ✅ PASSED |
| Func — compiled model supported list | `ov_cpu_func_tests` | B5-mod | ✅ PASSED |
| Func — plugin default false | `ov_cpu_func_tests` | B2 | ✅ PASSED |
| Func — plugin set true | `ov_cpu_func_tests` | B3 | ✅ PASSED |
| Func — plugin set false | `ov_cpu_func_tests` | B4 | ✅ PASSED |
| Func — compiled model default false | `ov_cpu_func_tests` | B6 | ✅ PASSED |
| Func — compiled model set true | `ov_cpu_func_tests` | B7 | ✅ PASSED |
| Func — compiled model set false | `ov_cpu_func_tests` | B8 | ✅ PASSED |
| Func — infer no-throw with true | `ov_cpu_func_tests` | B9 | ✅ PASSED |
| Func — results match default | `ov_cpu_func_tests` | B10 | ✅ PASSED |
| Func — property interaction streams | `ov_cpu_func_tests` | B11 | ✅ PASSED (both true/false infer, output tensors match) |
| **Total** | | **17** | ✅ **17/17 PASSED** |
