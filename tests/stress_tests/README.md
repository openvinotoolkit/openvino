# Stress Tests Suite

This test suite contains tests evaluating the behavior of various OpenVINO use
cases under stress conditions:

- MemCheckTests measuring memory required for the use cases and fail when memory
usage exceeds a pre-defined level.

- StressMemLeaksTests ensure that the use cases does not increase memory levels
when executing continuously.

- StressUnitTests executing various OpenVINO use cases in parallel
threads and processes.

StressUnitTests also covers coordinated runtime lifecycle scenarios:

| Test | Coverage |
|---|---|
| `stress_load_unload` | Repeated compile, infer, and unload |
| `stress_parallel_infer` | Parallel inference requests sharing one compiled model |
| `stress_concurrent_load_infer` | Model compilation concurrent with active inference |
| `stress_import_export` | Compiled-model export, import, and inference |
| `stress_mid_flight_cancel` | Inference cancellation while other requests are active |
| `stress_memory_pressure` | Simultaneous compiled models creating device memory pressure |
| `stress_destroy_compiled_model` | Compiled-model handle destruction while an infer request remains active |
| `stress_multiple_cores` | Simultaneous creation and destruction of multiple `ov::Core` instances |

The scenarios use public OpenVINO Runtime APIs and are device-independent. Add
`CPU`, `GPU`, or `NPU` to the configuration's `devices` section to run them on
an installed accelerator plugin. Export/import is skipped when a plugin does
not advertise the `EXPORT_IMPORT` capability.

Each test refers to configuration files located in `<test dir>\.automation`
folder. 

## Getting Started

Stress tests are based on the googletest framework. You can filter tests with
`--gtest_filter` and explore tests available with `--gtest_list_tests` options.

Tests measuring memory have a temporary limitation - those need to be executed
one at a time to mitigate memory statistics pollution. You can use
[gtest-parallel][gtest-parallel] for massive tests execution.

### Pre-requisites

- [gtest-parallel][gtest-parallel] to execute tests.

- An OpenVINO build or installation containing each plugin named in the test
configuration.

The coordinated runtime scenarios support Linux and Android. Process-level
parallelism uses `fork()` followed by `exec()` of a fresh `StressUnitTests`
child process, and scenario-level concurrency uses `std::thread`.
Memory measurement tests read `/proc/self/status` and are therefore separate
from the coordinated runtime scenarios.

### Building Tests

To build the tests, you need to have OpenVINO™ installed or build from source.
Before build the tests, open a terminal, set OpenVINO™ environment, and after that
run the commands below:
``` bash
source <OpenVINO_install_dir>/setupvars.sh
mkdir build && cd build
cmake .. && cmake --build . -j8
```

For Android, configure with the OpenVINO Android toolchain and deploy
`StressUnitTests`, its runtime libraries, a resolved test configuration, and
model files to the device.

### Preparing Test Data

Stress tests may work with models from [Open Model Zoo][open_model_zoo]. To use it, 
download and convert models to IRs using `./scripts/get_testdata.py` script.
Script will update test config file with data required for OMZ models execution.

The checked-in configurations are templates. Run `get_testdata.py` before test
discovery so each model entry contains resolved `path` and `full_path`
attributes.

From Intel network you can use models from cache at `vdp_tests` file share.
Refer to [VDP shared folders][VDP-shared-folders] on using file shares.

### Running Tests

``` bash
gtest-parallel <openvino_bin>/StressMemLeaksTests
```

Run only the coordinated scenarios with:

``` bash
StressUnitTests --test_conf=<test_conf_path> \
	--gtest_filter='StressUnitTests/UnitTestSuite.stress_*'
```

The accelerator is selected by the `<devices>` section of the resolved test
configuration. Use a configuration containing only the accelerator under test:

``` xml
<devices>
	<value>CPU</value>
</devices>
```

Run the complete coordinated scenario set on CPU:

``` bash
StressUnitTests --test_conf=<cpu_test_config.xml> \
	--gtest_filter='StressUnitTests/UnitTestSuite.stress_*'
```

Run the complete coordinated scenario set on GPU:

``` bash
StressUnitTests --test_conf=<gpu_test_config.xml> \
	--gtest_filter='StressUnitTests/UnitTestSuite.stress_*'
```

Run the complete coordinated scenario set on NPU:

``` bash
StressUnitTests --test_conf=<npu_test_config.xml> \
	--gtest_filter='StressUnitTests/UnitTestSuite.stress_*'
```

In the GPU and NPU configurations, replace `CPU` in the XML example with `GPU`
or `NPU`, respectively. The corresponding OpenVINO plugin must be installed and
available on the system.

Run one scenario by replacing `<test_name>` with a name from the table above:

``` bash
StressUnitTests --test_conf=<test_conf_path> \
	--gtest_filter='StressUnitTests/UnitTestSuite.<test_name>/*'
```

For example, run only concurrent `ov::Core` creation and destruction on NPU:

``` bash
StressUnitTests --test_conf=<npu_test_config.xml> \
	--gtest_filter='StressUnitTests/UnitTestSuite.stress_multiple_cores/*'
```

For MemCheckTests preferable way is:
``` bash
python ./scripts/run_memcheck.py --gtest_parallel <gtest_parallel_py_path> 
<openvino_bin>/MemCheckTests -- --test_conf=<test_conf_path> --refs_conf=<refs_conf_path>
``` 

MemCheckTests logs can be used to gather reference values based on current
memory consumption:

``` bash
mkdir -p MemCheckTests-logs && \
gtest-parallel -d ./MemCheckTests-logs ./MemCheckTests && \
grep -rh ./MemCheckTests-logs -e ".*<model " | sed -e "s/.*<model /<model /" | sort
```

[VDP-shared-folders]: https://wiki.ith.intel.com/display/DLSDK/VDP+shared+folders
[gtest-parallel]: https://github.com/google/gtest-parallel
[open_model_zoo]: https://github.com/openvinotoolkit/open_model_zoo