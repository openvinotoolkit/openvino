# Stress Tests Suite

This test suite contains tests evaluating the behavior of various OpenVINO use
cases under stress conditions:

- MemCheckTests measuring memory required for the use cases and fail when memory
usage exceeds a pre-defined level.

- StressMemLeaksTests ensure that the use cases does not increase memory levels
when executing continuously.

- StressUnitTests executing various Inference Engine use cases in parallel
threads and processes.

Each test refers to configuration files located in `<test dir>\.automation`
folder. 

## Getting Started

Stress tests are based on the googletest framework. You can filter tests with
`--gtest_filter` and explore tests available with `--gtest_list_tests` options.

Tests measuring memory have a temporary limitation - those need to be executed
one at a time to mitigate memory statistics pollution. You can use
[gtest-parallel][gtest-parallel] for massive tests execution.

### Pre-requisites

- Linux OS to build the tests.

- [gtest-parallel][gtest-parallel] to execute tests.

### Building Tests

To build the tests, you need to have OpenVINO™ installed or build from source.
Before build the tests, open a terminal, set OpenVINO™ environment, and after that
run the commands below:
``` bash
source <OpenVINO_install_dir>/setupvars.sh
mkdir build && cd build
cmake .. && cmake --build . -j8
```

### Preparing Test Data

Stress tests may work with models from [Open Model Zoo][open_model_zoo]. To use it, 
download and convert models to IRs using `./scripts/get_testdata.py` script.
Script will update test config file with data required for OMZ models execution.

From Intel network you can use models from cache at `vdp_tests` file share.
Refer to [VDP shared folders][VDP-shared-folders] on using file shares.

### Running Tests

``` bash
gtest-parallel <openvino_bin>/StressMemLeaksTests
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