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
* `--plugin_lib_name` is name of plugin library. The example is MKLDNNPlugin. Use only with unregistered in IE Core devices.
* `--disable_test_config` allows to ignore all skipped tests with the exception of `DISABLED_` prefix using.
* `--skip_config_path` allows to specify paths to files contain regular expressions list to skip tests.
* `--extend_report` allows not to re-write device results to the report (add results of this run to the existing). Mutually exclusive with --report_unique_name.
* `--report_unique_name` allows to save report with unique name (report_pid_timestamp.xml). Mutually exclusive with --extend_report.
* `--save_report_timeout` allows to try to save report in cycle using timeout (in seconds).
* `--output_folder` Paths to the output folder to save report.
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
> python3 gtest_parallel.py /path/to/openvino/bin/intel64/Debug/conformanceTests -d . 
> --gtest_filter=*Add*:*BinaryConv* -- --input_folders=/path/to/ir_1,/path/to/ir_2 --device=CPU 
> --report_unique_name --output_folder=/path/to/temp_output_report_folder
> ```
> All arguments after `--` symbol is forwarding to `conformanceTests` target.
> 
> After using `--report_unique_name` argument please run
> [the merge xml script](./../../../../ie_test_utils/functional_test_utils/layer_tests_summary/merge_xmls.py) 
> to aggregate the results to one report.
> The example of usage is:
> ```
> python3 merge_xmls.py --input_folders=/path/to/temp_output_report_folder --output_folder=/path/to/output_report_folder --output_filename=report_aggregated
> ```

## How to build operation coverage report
Run [the script](./../../../../ie_test_utils/functional_test_utils/layer_tests_summary/summarize.py) to generate `html` report.
The example of using the script is:
```
python3 summarize.py --xml /opt/repo/infrastructure-master/thirdparty/gtest-parallel/report.xml --out /opt/repo/infrastructure-master/thirdparty/gtest-parallel/
```