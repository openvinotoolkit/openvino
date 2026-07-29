# AppVerifier Tests Suite

This test suite is used to detect whether AppVerifier will report a memory leak.

## Getting Started

AppVerifier tests are based on the googletest framework. You can filter tests with
`--gtest_filter` and explore tests available with `--gtest_list_tests` options.

### Pre-requisites

- Windows OS to build the tests.

### Building Tests

To build the tests, you need to have OpenVINO™ installed or build from source.
Before build the tests, open a terminal, set OpenVINO™ environment, and after that
run the commands below:
``` bash
<OpenVINO_install_dir>/setupvars.bat
mkdir build && cd build
cmake .. && cmake --build . --config Release -j8
```

### Running Tests

``` bash
.\test\Release\ov_appverifier_tests.exe
```

This test can be run directly using the above command, but if you want to detect whether there is a memory leak, the test executable file need to be added in AppVerifier (refer to https://learn.microsoft.com/en-us/windows-hardware/drivers/devtest/application-verifier-testing-applications) before running the above command.
