# Conformance test runner

## Description
Conformance suit is a set of tests with parameters independent from plug-in specific and limitations. It contains:
* `ReadIR`. Allow to read IRs from folders recursive, infer it and compare results with reference.
* `OpImplCheckTest`. Allow to check operation plugin implementation status (`Implemented`/`Not implemented`).

  **NOTE**: This test suite is in active development. The implementation status based on all other test results.
* 

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
   
## How to generate Conformance operation set
Run the following commands:
1. Clone [`Open Model Zoo repo`](https://github.com/openvinotoolkit/open_model_zoo)
2. Download all possible models using [Downloader tool](https://github.com/openvinotoolkit/open_model_zoo/blob/master/tools/model_tools/downloader.py) from 
   the repo.
3. Convert downloaded models to IR files using [Converter tool](https://github.com/openvinotoolkit/open_model_zoo/blob/master/tools/model_tools/converter.py) from the repo.
4. Run [Subgraph dumper](./../subgraphs_dumper/README.md) to collect unique operation set from the models.
> **NOTE**:
> 
> The conformance suite run in internal CI checks is based on `2021.4` release. **Please, use `Model optimizer`, 'Subgraph dumper' and `Open Model Zoo` tools 
> from `2021.4`.**

## How to run operation conformance suite
The target is able to take the following command-line arguments:
* `-h` prints target command-line options with description.
* `--device` specifies target device.
* `--input_folders` specifies the input folders with IRs or '.lst' file contains IRs path. Delimiter is `,` symbol.
* `--plugin_lib_name` is name of plugin library. The example is `openvino_intel_cpu_plugin`. Use only with unregistered in IE Core devices.
* `--disable_test_config` allows to ignore all skipped tests with the exception of `DISABLED_` prefix using.
* `--skip_config_path` allows to specify paths to files contain regular expressions list to skip tests. [Examples](./op_conformance_runner/skip_configs)
* `--config_path` allows to specify path to file contains plugin config. [Example](./op_conformance_runner/config/config_example.txt)
* `--extend_report` allows not to re-write device results to the report (add results of this run to the existing). Mutually exclusive with --report_unique_name.
* `--report_unique_name` allows to save report with unique name (report_pid_timestamp.xml). Mutually exclusive with --extend_report.
* `--save_report_timeout` allows to try to save report in cycle using timeout (in seconds).
* `--output_folder` Paths to the output folder to save report.
* `--extract_body` allows to count extracted operation bodies to report.
* `--shape_mode` Optional. Allows to run `static`, `dynamic` or both scenarios. Default value is empty string allows to run both scenarios. Possible values 
  are `static`, `dynamic`, ``
* `--test_timeout` Setup timeout for each test in seconds, default timeout 900seconds (15 minutes).
* All `gtest` command-line parameters

The result of execution is `report.xml` file. It demonstrates tests statistic like pass rate, passed, crashed, skipped failed tests and plugin implementation 
per 
operation for 
devices.

> **NOTE**:
> 
> Using of GTest parallel tool to run `conformanceTests` helps to report crashed tests and collect correct statistic 
> after unexpected crashes. 
> 
> Use [`Gtest parallel`](https://github.com/google/gtest-parallel) from official repository with [this fix](https://github.com/google/gtest-parallel/pull/76).
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

## How to create operation conformance report
Run [the script](./../../../../ie_test_utils/functional_test_utils/layer_tests_summary/summarize.py) to generate `html` report.
The example of using the script is:
```
python3 summarize.py --xml /opt/repo/infrastructure-master/thirdparty/gtest-parallel/report.xml --out /opt/repo/infrastructure-master/thirdparty/gtest-parallel/
```
> **NOTE**:
>
> Please, do not forget to copy [styles folder](./../../../../ie_test_utils/functional_test_utils/layer_tests_summary/template) to the output directory. It 
> helps to provide report with the filters and other usable features.

Report contains statistic based on conformance results and filter fields at the top of the page.