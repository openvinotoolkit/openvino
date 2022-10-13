# OpenVINO IR Frontend tests

OpenVINO IR tests covers the next frontend functionality: parser IRs, read model from the memory, extensions support.

## How to build and run tests

CMake target `ov_ir_frontend_tests` is responsible for building IR tests. IR Frontend tests uses `gtest` framework for tests execution.

After the build `ov_ir_frontend_tests` (`ov_ir_frontend_tests.exe` for Windows) binary files can be used to run all tests for the IR frontend.

If you would like to get a tests coverage report for IR frontend, please read the page about [how to measure coverage](../../../../docs/dev/test_coverage.md)

## See also

 * [OpenVINO IR Frontend README](../README.md)
 * [OpenVINO™ README](../../../../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)
