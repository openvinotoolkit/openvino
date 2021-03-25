# Conformance test runner

## Description
Conformance suit is a set of tests with parameters independent from plug-in specific and limitations. It contains:
* `ReadIR`. Allow to read IRs from folders recursive, infer it and compare results with reference.

## How to build
Run the following command in build directory:
1. Generate CMake project:
   ```
   cmake -DENABLE_FUNCTIONAL_TESTS=ON ..
   ```
2. Build the target:
   ```
   make conformanceTests
   ```
   
## How to run
The target is able to take the following command-line arguments:
* `-h` prints target command-line options with description.
* `--device` specifies target device.
* `--input_folders` specifies folders with IRs to run. The separator is `,`.
* `--disable_test_config` allows to ignore all skipped tests with the exception of `DISABLED_` prefix using.
* `--extend_report` allows not to re-write device results to the report (add results of this run to the existing).
* All `gtest` command-line parameters

The result of execution is `report.xml` file. It demonstrates tests statistic like pass rate, passed, crashed, skipped and failed tests per operation for 
devices.

> **NOTE**:
> 
> Using of GTest parallel tool to run `conformanceTests` helps to report crashed tests and collect correct statistic 
> after unexpected crashes. 
> 
> The example of usage is:
> ```
> python3 gtest_parallel.py /opt/repo/openvino/bin/intel64/Debug/conformanceTests -d . --gtest_filter=*1613473581844763495*:*roi_align*:*PSROIPooling*:*Add*:*BinaryConv* -- --input_folders=/opt/repo/roi_align,/opt/repo/omz/out --device=CPU
> ```
> All arguments after `--` symbol is forwarding to `conformanceTests` target.

## How to build operation coverage report
Run [the script](./../../../../ie_test_utils/functional_test_utils/layer_tests_summary/summarize.py) to generate `html` report.
The example of using the script is:
```
python3 summarize.py --xml /opt/repo/infrastructure-master/thirdparty/gtest-parallel/report.xml --out /opt/repo/infrastructure-master/thirdparty/gtest-parallel/
```