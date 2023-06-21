# AUTO Plugin tests

## AUTO plugin Unit Tests

Auto unit test is a set of unit tests using gmock, each of which is for testing selection logic, device list parser, meta device filter and all other internal functions of Auto plugin.

## How to run tests

### Build unit test

1. Turn on `ENABLE_TESTS` in cmake option:

   ```bash
   cmake -DCMAKE_BUILD_TYPE=Release \
       -DENABLE_TESTS=ON \
   ```

2. Build

   ```bash
   make ov_auto_unit_tests
   ```

3. You can find `ov_auto_unit_tests` in *bin* directory after build

### Run unit test

You can run _`ov_auto_unit_tests`_ in *bin* directory which is the output of OpenVINO build

If you want to run a specific unit test, you can use `gtest_filter` option as follows:

```
./ov_auto_unit_tests --gtest_filter='*filter_name*'
```

Then, you can get the result similar to:

```bash
openvino/bin/intel64/Release$ ./ov_auto_unit_tests --gtest_filter=*AutoReleaseHelperTest*cpuLoadFailure_accelerateorLoadFailure*
Running main() from /home/openvino/thirdparty/gtest/gtest/googletest/src/gtest_main.cc
Note: Google Test filter = *AutoReleaseHelperTest*cpuLoadFailure_accelerateorLoadFailure*
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from smoke_Auto_BehaviorTests/AutoReleaseHelperTest
[ RUN      ] smoke_Auto_BehaviorTests/AutoReleaseHelperTest.releaseResource/cpuLoadFailure_accelerateorLoadFailure
[       OK ] smoke_Auto_BehaviorTests/AutoReleaseHelperTest.releaseResource/cpuLoadFailure_accelerateorLoadFailure (732 ms)
[----------] 1 test from smoke_Auto_BehaviorTests/AutoReleaseHelperTest (732 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (732 ms total)
[  PASSED  ] 1 test.
```