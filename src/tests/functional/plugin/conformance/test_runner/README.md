# Conformance test runner

## Description
Conformance suites certify plugin functionality using a set of tests with plugin specificity independent parameters. There are two types of conformance validation.

### API Conformance
The suite checks the following OpenVINO API entities in a plugin implementation:
* plugin
* compiled model (executable network)
* infer request
Also, there are test instantiations to validate hardware plugin functionality via software plugins (for example, MULTI, HETERO, etc.) for the entities.

The other part of the API conformance suite is QueryModel validation:
* `ReadIR_queryModel` tests validate the `query_model` API using a simple single operation graph (Conformance IR) based on model parameters.
* `OpImplCheck` tests are simple synthetic checks to `query_model` and set implementation status for each operation.

A result of the `apiConformanceTests` run is two xml files: `report_api.xml` and `report_opset.xml`. The first one shows OpenVINO API entities' test statistics for each OpenVINO API entity, such as passed/failed/crashed/skipped/hanging, tests number, pass rates, and implementation status. The second one demonstrates the `query_model` results for each operation.



### Opset Conformance
The suite validates an OpenVINO operation plugin implementation, using simple single operation graphs (Conformance IR) taken from models. The plugin inference output is compared with the reference.

 The suite contains:
* `ReadIR_compareWithRefs` set allows reading IRs from folders recursively, inferring them, and comparing plugin results with the reference.
* `OpImplCheckTest` set checks an operation plugin implementation status, using a simple synthetic single operation graph (`Implemented`/`Not implemented`). The suite checks only `compile_model` without  comparison with the reference.

A result of the `conformanceTests` run is the `report_opset.xml` file. It shows tests statistic, like pass rate, passed, crashed, skipped, failed tests, and plugin implementation per operation for devices.

## How to build
Run the following command in build directory:
1. Generate CMake project:
   ```
   cmake -DENABLE_TESTS=ON -DENABLE_FUNCTIONAL_TESTS=ON ..
   ```
2. Build the targets:
   ```
   make --jobs=$(nproc --all) subgraphsDumper
   make --jobs=$(nproc --all) conformanceTests
   make --jobs=$(nproc --all) apiConformanceTests
   ```
3. Build plugins to validate:
   ```
   make --jobs=$(nproc --all) lib_plugin_name
   ```
   
## How to generate Conformance IRs set
Run the following commands:
1. Clone [`Open Model Zoo repo`](https://github.com/openvinotoolkit/open_model_zoo)
2. Download all models using [Downloader tool](https://github.com/openvinotoolkit/open_model_zoo/blob/master/tools/model_tools/downloader.py) from the repo.
3. Convert downloaded models to IR files using [Converter tool](https://github.com/openvinotoolkit/open_model_zoo/blob/master/tools/model_tools/converter.py) from the repo.
4. Run [Subgraph dumper](./../subgraphs_dumper/README.md) to collect unique operation set from the models.



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

> **NOTE**:
> 
> Using of `GTest parallel` tool to run a conformance suite helps to report crashed tests and collect correct statistic after unexpected crashes. 
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
>  If you use the `--report_unique_name` argument, run
> [the merge xml script](./../../../../ie_test_utils/functional_test_utils/layer_tests_summary/merge_xmls.py) 
> to aggregate the results to one xml file. Check command-line arguments with `--help` before running the command.
> The example of usage is:
> ```
> python3 merge_xmls.py --input_folders=/path/to/temp_output_report_folder --output_folder=/path/to/output_report_folder --output_filename=report_aggregated
> ```

## How to create operation conformance report
Run [the summarize script](./../../../../ie_test_utils/functional_test_utils/layer_tests_summary/summarize.py) to generate `html` and `csv` report. Check command-line arguments with `--help` before running the command.
The example of using the script is:
```
python3 summarize.py --xml /opt/repo/infrastructure-master/thirdparty/gtest-parallel/report.xml --out /opt/repo/infrastructure-master/thirdparty/gtest-parallel/
```
> **NOTE**:
>
> Please, do not forget to copy [styles folder](./../../../../ie_test_utils/functional_test_utils/layer_tests_summary/template) to the output directory. It 
> helps to provide report with the filters and other usable features.

The report contains statistics based on conformance results and filter fields at the top of the page.

# [Simple conformance runner](./../../../../ie_test_utils/functional_test_utils/layer_tests_summary/run_conformance.py)
There is a simple python runner to complete the whole conformance pipeline locally. Some steps could be excluded from the pipeline by command-line parameter configuration.

## The conformance pipeline steps:
1. (Optional) Download OMZ models.
2. (Optional) Convert models to IR or prepare a folder with IRs.
3. (Optional) Run `SubgraphDumper` to generate a simple single op graph based on models or download the `conformance_ir` folder.
4. Run conformance test executable files.
5. Generate conformance reports.

## Command-line arguments
The script has the following arguments:
* `-h, --help`          show this help message and exit
* `-m MODELS_PATH, --models_path MODELS_PATH`
                        Path to the directory/ies containing models to dump subgraph (the default way is to download OMZ). If `--s=0`, specify the Conformance IRs directory
* `-d DEVICE, --device DEVICE`
                        Specify the target device. The default value is CPU
* `-ov OV_PATH, --ov_path OV_PATH`
                        OV binary files path. The default way is trying to find an installed OV by `INTEL_OPENVINO_DIR` in the environment variables or to find the absolute path of the OV repo by using the script path
* `-w WORKING_DIR, --working_dir WORKING_DIR`
                        Specify a working directory to save all artifacts, such as reports, models, conformance_irs, etc.
* `-t TYPE, --type TYPE`
                        Specify conformance type: `OP` or `API`. The default value is `OP`
* `-s DUMP_CONFORMANCE, --dump_conformance DUMP_CONFORMANCE`
                        Set '1' if you want to create Conformance IRs from custom/downloaded models. In other cases, set `0`. The default value is '1'

> **NOTE**:
> All arguments are optional and have default values to reproduce OMZ conformance results in a default way.

## Examples of usage:
1. Use the default way to reproduce opset conformance results for OMZ on GPU:
```
python3 run_conformance.py -d GPU
``` 
2. Use the conformance pipeline to check new models support (as IRs) on the CPU plugin and save results to a custom directory:
```
python3 run_conformance.py -m /path/to/new/model_irs -s=1 -w /path/to/working/dir -d CPU
``` 
3. Use custom OV build to check GNA conformance using pre-generated conformance_irs:
```
python3 run_conformance.py -m /path/to/conformance_irs -s=0 -ov /path/to/ov_repo_on_custom_branch -d GNA
``` 

> **IMPORTANT NOTE:**
> If you need to debug some conformance tests, use the binary run as the default method. If you want to get conformance results or reproduce CI behavior, use the simple python runner.

## See also
 * [OpenVINO™ README](../../../../../../README.md)
 * [OpenVINO Core Components](../../../../../README.md)
 * [Developer documentation](../../../../../../docs/dev/index.md)