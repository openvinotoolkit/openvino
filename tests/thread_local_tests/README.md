# Thread_local Tests Suite

This test suite is used to detect whether AppVerifier will report a memory leak caused by thread_local.

## Getting Started

Thread_local tests are based on the googletest framework. You can filter tests with
`--gtest_filter` and explore tests available with `--gtest_list_tests` options.

### Pre-requisites

- Windows OS to build the tests.

### Building Tests

To build the tests, you need to have OpenVINO™ installed or build from source.
Before build the tests, open a terminal, set OpenVINO™ environment, and after that
run the commands below:
``` bash
source <OpenVINO_install_dir>/setupvars.sh
mkdir build && cd build
cmake .. && cmake --build . -j8
```

### Running Tests

``` bash
\test\Release\ov_thread_local_test.exe
```
