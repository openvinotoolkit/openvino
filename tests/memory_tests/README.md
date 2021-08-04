# Memory Tests

This test suite contains pipelines, which are executables. 
Memory tests measuring memory required for the use cases and fail when memory
usage exceeds a pre-defined level.

## Prerequisites

To build the memory tests, you need to have OpenVINO™ installed or build from source.

## Measure Time

To build and run the tests, open a terminal, set OpenVINO™ environment and run
the commands below:

1. Build tests:
``` bash
mkdir build && cd build
cmake .. && make memory_tests
```

If you don't have OpenVINO™ installed you need to have the `build` folder, which
is created when you configure and build OpenVINO™ from sources:

``` bash
cmake .. -DInferenceEngine_DIR=$(realpath ../../../build) && make memory_tests
```
For old versions of OpenVINO™ from sources use `-DInferenceEngineDeveloperPackage_DIR`:
``` bash
cmake .. -DInferenceEngineDeveloperPackage_DIR=$(realpath ../../../build) && make memory_tests
```

2. Run test:
``` bash
./utils/scripts/run_test.py ../../bin/intel64/Release/memorytest_infer -m model.xml -d CPU
```

2. Run several configurations using `pytest`:
``` bash
pytest ./utils/test_runner/test.py --exe ../../bin/intel64/Release/memorytest_infer

# For parse_stat testing:
pytest ./utils/scripts/run_test.py
```
