# Memory Tests

This test suite contains pipelines, which are executables. 
Memory tests measuring memory required for the use cases and fail when memory
usage exceeds a pre-defined level.

## Prerequisites

To build memory tests, you need to have OpenVINO™ installed or build from source.

## Measure Time

To build and run the tests, open a terminal, set OpenVINO™ environment and run
the commands below:

1. Build tests:
``` bash
mkdir build && cd build
cmake .. && make memory_tests
```

2. Install tests:
``` bash
сmake install <build_dir> --prefix <install_path>
```

3. Run test:
``` bash
./scripts/run_memorytest.py <install_path>/tests/memtest_infer -m model.xml -d CPU
```

4. Run several configurations using `pytest`:
``` bash
pytest ./test_runner/test.py --exe <install_path>/tests/memorytest_infer
# For parse_stat testing:
pytest ./scripts/run_memorytest.py
```