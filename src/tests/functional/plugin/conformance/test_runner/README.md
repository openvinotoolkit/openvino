# Conformance Test Runner

## Description

Conformance suites certify plugin functionality using a set of tests with plugin specificity independent parameters. There are two types of conformance validation.

### API Conformance

The suite checks the following OpenVINO API entities in a plugin implementation:
* plugin
* compiled model (executable network)
* infer request
Also, there are test instantiations to validate hardware plugin functionality via software plugins (for example, MULTI, HETERO, etc.) for the entities.

API conformance contains 2 test types:
* `mandatory` scope validates functionality required to be implemented in plugin. The expected pass-rate of mandatory tests is 100% for conformant harware plugin.
* `optional` scope validates both not key functionality in an OpenVINO plugin. OpenVINO provides API, but plugin is able to decide not to implement these methods. Optional scope contains all API Conforformance tests.

>**NOTE:** To run only mandatory API Conformance tests use `--gtest_filter=*mandatory*`.

A result of the `ov_api_conformance_tests` run is report `report_api.xml`. It shows OpenVINO API entities' test statistics for each OpenVINO API entity, such as `passed/failed/crashed/skipped/hanging`, tests number, pass rates, and implementation status.

### Opset Conformance

The other part of the API conformance suite is QueryModel validation:

The suite contains:
* `ReadIR_Inference` set allows reading model based graphs from folders recursively, inferring them, and comparing plugin results with the reference.
* `ReadIR_QueryModel` tests validate the `query_model` API (support the operation by plugin), using a simple ]graph (Conformance IR) extracted from models.
* `ReadIR_ImportExport` tests exports and imports of compiled model, using a graph (Conformance IR) based on model parameters.
* `OpImplCheckTest` set checks an operation plugin implementation status, using a simple synthetic single operation graph (`Implemented`/`Not implemented`). The suite checks only `compile_model` without comparison with the reference.

A result of the `ov_op_conformance_tests` run is the `report_opset.xml` file. It shows tests statistic, like pass rate, passed, crashed, skipped, failed tests, and plugin implementation per operation for devices.

## How to build

Run the following commands in the build directory:
1. Generate CMake project:
   ```
   cmake -DENABLE_TESTS=ON -DENABLE_FUNCTIONAL_TESTS=ON ..
   ```
2. Build the targets:
   ```
   make --jobs=$(nproc --all) ov_subgraphs_dumper
   make --jobs=$(nproc --all) ov_op_conformance_tests
   make --jobs=$(nproc --all) ov_api_conformance_tests
   ```
3. Build plugins to validate:
   ```
   make --jobs=$(nproc --all) lib_plugin_name
   ```
4. Install python dependencies to use `run_conformance` script:
   ```
   cd /path/to/openvino/src/tests/test_utils/functional_test_utils/layer_tests_summary
   pip3 install -r [requirements.txt](./../../../../../tests/test_utils/functional_test_utils/layer_tests_summary/requirements.txt)
   ```

## How to run using [simple conformance runner](./../../../../../tests/test_utils/functional_test_utils/layer_tests_summary/run_conformance.py)

There is a simple python runner to complete the whole conformance pipeline locally. Some steps could be excluded from the pipeline by command-line parameter configuration.

>NOTE: Conformance reports `ov python api` WARNING in case of its absence. `ov python api` is not required to get a conformance results. It is a way to get HASHED conformance IR names after `ov_subgraphs_dumper` tool using (in case of `-s=1`).

### The conformance pipeline steps:

1. (Optional: Applicable only for Opset Conformance suite) Download models/conformance IR via URL / copy archive to working directory / verify dirs / check list-files.
2. (Optional: Applicable only for Opset Conformance suite) Run `ov_subgraphs_dumper` to extract graph from models or download the `conformance_ir` folder. (if `-s=1`)
3. Run conformance test executable files.
4. Generate conformance reports.

### Command-line arguments

The script has the following optional arguments:
* `h, --help`           Show this help message and exit
* `d DEVICE, --device DEVICE`
                        Specify a target device. The default value is `CPU`
* `t TYPE, --type TYPE` 
                        Specify conformance type: `OP` or `API`. The default value is `OP`
* `-gtest_filter GTEST_FILTER`
                        Specify gtest filter to apply for a test run. E.g. *Add*:*BinaryConv*. The default value is `None`
* `w WORKING_DIR, --working_dir WORKING_DIR`
                        Specify a working directory to save a run artifacts. The default value is `current_dir/temp`.
* `m MODELS_PATH, --models_path MODELS_PATH`
                        Path to the directory/ies containing models to dump subgraph (the default way is to download conformance IR). It may be directory, archieve file, .lst file with models to download by a link, model file paths. If --s=0, specify the Conformance IRs directory. NOTE: Applicable only for Opset Conformance.
* `ov OV_PATH, --ov_path OV_PATH`
                        OV binary path. The default way is to find the absolute path of latest bin in the repo (by using script path)
* `j WORKERS, --workers WORKERS`
                        Specify number of workers to run in parallel. The default value is `CPU_count`
* `c OV_CONFIG_PATH, --ov_config_path OV_CONFIG_PATH`
                        Specify path to a plugin config file as `.lst` file. Default value is ``
* `s DUMP_GRAPH, --dump_graph DUMP_GRAPH`
                        Set '1' to create Conformance IRs from models using ov_subgraphs_dumper tool. The default value is '0'.
                        NOTE: Applicable only for Opset Conformance.
* `sm SPECIAL_MODE, --special_mode SPECIAL_MODE`
                        Specify shape mode (`static`, `dynamic` or ``) for Opset conformance or API scope type (`mandatory` or ``). Default value is ``
*  `-e ENTITY, --entity ENTITY`
                        Specify validation entity: `Inference`, `ImportExport`, `QueryModel` or `OpImpl` for `OP` or `ov`. Default value is `ov_compiled_model`, `ov_infer_request` or `ov_plugin` for `API`. Default value is ``(all)
* `p PARALLEL_DEVICES, --parallel_devices PARALLEL_DEVICES`
                        Parallel over HW devices. For example run tests over `GPU.0` and `GPU.1` in case when device are the same
* `f EXPECTED_FAILURES, --expected_failures EXPECTED_FAILURES`
                        Excepted failures list file path as csv. See more in the [Working with expected failures](#working-with-expected-failures) section.
* `u EXPECTED_FAILURES_UPDATE, --expected_failures_update EXPECTED_FAILURES_UPDATE`
                        Overwrite expected failures list in case same failures were fixed
* `-cache_path CACHE_PATH`
                        Path to the cache file with test_name list sorted by execution time as `.lst` file!
* `-r DISABLE_RERUN, --disable_rerun DISABLE_RERUN`
                        Disable re-run of interapted/lost tests. Default value is `False`
* `--timeout TIMEOUT`
                        Set a custom timeout per worker in s

> **NOTE**: All arguments are optional and have default values to reproduce OMZ based Opset conformance results  on `CPU` in a default method.

> **NOTE**: The approach can be used as custom model scope validator!

## Examples of usage:

1. Use the default method to reproduce opset conformance results for OMZ on GPU:
```
python3 run_conformance.py -d GPU
```
2. Use the conformance pipeline to check new models support (as IRs) on the CPU plugin and save results to a custom directory:
```
python3 run_conformance.py -m /path/to/new/model_irs -s=1 -w /path/to/working/dir -d CPU
```
3. Use custom OV build to check GNA conformance, using pre-generated `conformance_irs`:
```
python3 run_conformance.py -m /path/to/conformance_irs -s=0 -ov /path/to/ov_repo_on_custom_branch/bin/intel64/Debug -d GNA
```
4. Use the default method to reproduce opset conformance results for OMZ on TEMPLATE using custom config file:
```
python3 run_conformance.py -d GPU -c /path/to/ov/config.lst
```

> **IMPORTANT NOTE:** If you need to debug some conformance tests, use the binary run as the default method. If you want to get conformance results or reproduce CI behavior, use the simple python runner.

## How to generate Conformance IRs set

Run the following commands:
1. Clone [`Open Model Zoo repo`](https://github.com/openvinotoolkit/open_model_zoo) or prepare custom model scope
2. Download all models using [Downloader tool](https://github.com/openvinotoolkit/open_model_zoo/blob/master/tools/model_tools/downloader.py) from the repo.
3. Convert downloaded models to IR files, using [Converter tool](https://github.com/openvinotoolkit/open_model_zoo/blob/master/tools/model_tools/converter.py) from the repo.
4. Run [Subgraph dumper](./../subgraphs_dumper/README.md) to collect unique operation set from the models.

> **NOTE:** There are ready to use Conformance IRs based on latest OMZ scope in [the public storage](https://storage.openvinotoolkit.org/test_data/conformance_ir/conformance_ir.tar).

> **NOTE:** You can use the above algorithm to generate Conformance IRs using a custom model scope supported by OpenVINO.

## How to run a conformance suite

The target is able to take the following command-line arguments:
* `-h` prints target command-line options with description.
* `--device` specifies target device.
* `--input_folders` specifies the input folders with IRs or `.lst` file. It contains paths, separated by a comma `,`.
* `--plugin_lib_name` is a name of a plugin library. The example is `openvino_intel_cpu_plugin`. Use only with unregistered in OV Core devices.
* `--disable_test_config` allows ignoring all skipped tests with the exception of `DISABLED_` prefix using.
* `--skip_config_path` allows specifying paths to files. It contains a list of regular expressions to skip tests. [Examples](./op_conformance_runner/skip_configs/skip_config_example.lst)
* `--config_path` allows specifying the path to a file that contains plugin config. [Example](./op_conformance_runner/config/config_example.txt)
* `--extend_report` allows you not to re-write device results to the report (add results of this run to the existing one). Mutually exclusive with `--report_unique_name`.
* `--report_unique_name` allows you to save a report with a unique name (`report_pid_timestamp.xml`). Mutually exclusive with `--extend_report`.
* `--save_report_timeout` allows saving a report in the cycle, using timeout (in seconds).
* `--output_folder` specifies the path to the output folder to save a report.
* `--extract_body` allows you to count extracted operation bodies to a report.
* `--shape_mode` is optional. It allows you to run `static`, `dynamic` , or both scenarios. The default value is an empty string, which allows running both scenarios. Possible values
  are `static`, `dynamic`, ``
* `--test_timeout` specifies setup timeout for each test in seconds. The default timeout is 900 seconds (15 minutes).
* `--ignore_crash` Optional. Allow to not terminate the whole run after crash and continue execution from the next test. This is organized with custom crash handler. Please, note, that handler work for test body,  if crash happened on SetUp/TearDown stage, the process will be terminated.
* All `gtest` command-line parameters

> **NOTE**:
>
> Using [`parallel_runner`](./../../../../../tests/test_utils/functional_test_utils/layer_tests_summary/run_parallel.py) tool to run a conformance suite helps to report crashed tests and collect correct statistics after unexpected crashes.
> The tool is able to work in two modes:
> * one test is run in a separate thread (first run, as the output the cache will be saved as a custom file).
> * similar load time per one worker based on test execution time. May contain different test count per worker.
>
> The example of usage is:
> ```
> python3 run_parallel.py -e=/path/to/openvino/bin/intel64/Debug/conformanceTests -d .
> --gtest_filter=*Add*:*BinaryConv* -- --input_folders=/path/to/ir_1,/path/to/ir_2 --device=CPU
> --report_unique_name --output_folder=/path/to/temp_output_report_folder
> ```
> All arguments after `--` symbol is forwarding to `conformanceTests` target.
>
>  If you use the `--report_unique_name` argument, run
> [the merge xml script](./../../../../../tests/test_utils/functional_test_utils/layer_tests_summary/merge_xmls.py)
> to aggregate the results to one *xml* file. Check command-line arguments with `--help` before running the command.
> The example of usage is:
> ```
> python3 merge_xmls.py --input_folders=/path/to/temp_output_report_folder --output_folder=/path/to/output_report_folder --output_filename=report_aggregated
> ```

## Working with expected failures

The `run_conformace.py` script has an optional `--expected_failures` argument which accepts a path to a csv file with a list of tests that should not be run. 

You can find the files with the most up-to-date expected failures for different devices and conformance types [here](./../../../../../tests/test_utils/functional_test_utils/layer_tests_summary/skip_configs).

These files are used in [the Linux GitHub workflow](./../../../../../../.github/workflows/ubuntu_22.yml) for test skip. 

You can update the file(s) you need with either new passing tests, i.e., when something is fixed, or with new failing tests to skip them. The changes will be reflected in the GitHub actions pipeline, in the `Conformance_Tests` job.

## How to create a conformance report

Run [the summarize script](./../../../../../tests/test_utils/functional_test_utils/layer_tests_summary/summarize.py) to generate `html` and `csv` report. Check command-line arguments with `--help` before running the command.
The example of using the script is:
```
python3 summarize.py --xml /opt/repo/infrastructure-master/thirdparty/gtest-parallel/report_opset.xml --out /opt/repo/infrastructure-master/thirdparty/gtest-parallel/ -t OP
```
```
python3 summarize.py --xml /opt/repo/infrastructure-master/thirdparty/gtest-parallel/report_api.xml --out /opt/repo/infrastructure-master/thirdparty/gtest-parallel/ -t API
```
> **NOTE**: Remember to copy [styles folder](./../../../../../tests/test_utils/functional_test_utils/layer_tests_summary/template) to the output directory. It helps to provide a report with filters and other useful features.

The report contains statistics based on conformance results and filter fields at the top of the page.

## See Also

 * [OpenVINO™ README](../../../../../../README.md)
 * [OpenVINO Core Components](../../../../../README.md)
 * [Developer documentation](../../../../../../docs/dev/index.md)
