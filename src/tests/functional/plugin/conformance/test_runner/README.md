# Conformance test runner

## Description
Conformance suites certifies plugin functionality using a set of tests with plugin specificity independent parameters. There are 2 types of conformance validation.

### `API Conformance`
The suite checks OpenVINO API the folowing entities a plugin implementation (Also there are test instantiations to validate hardware plugin functionality via software plugins (e.g. `MULTI`, `HETERO` and etc.) for the entities):
* plugin
* compiled model (executable network)
* infer request

The other part of API conformance suite is QueryModel validation:
* `ReadIR_queryModel` tests validates `query_model` methd using simple single operation graph (Conformance IR) based on model parameters.
* `OpImplCheck` tests are simple syntetic checks to verify the method and set implementation status.

A result of `apiConformanceTests` run is 2 xml files: `report_api.xml` and `report_opset.xml`. The first file shows OpenVINO API entities results. The second one demonstration `query_model` results.



### `Opset Conformance`
The suite validates OpenVINO an operation plugin implemenatation using simple signgle op graph (Conformance IR) taken from models. The plugin results is compared with refrence. The suite contains:
*`ReadIR_compareWithRefs`. Allow to read IRs from folders recursive, infer it and compare plugin results with reference.
*`OpImplCheckTest`. Allow to check operation plugin implementation status to compile model (`Implemented`/`Not implemented`).  

A result of `conformanceTests` run is xml file `report_opset.xml` to show test results. It demonstrates tests statistic like pass rate, passed, crashed, skipped failed tests and plugin implementation per operation for devices.

## How to build
Run the following command in build directory:
1. Generate CMake project:
   ```
   cmake -DENABLE_FUNCTIONAL_TESTS=ON ..
   ```
2. Build the targets:
   ```
   make --jobs=$(nproc --all) subgraphsDumper
   make --jobs=$(nproc --all) conformanceTests
   make --jobs=$(nproc --all) apiConformanceTests
   ```
3. Build plugins to validate
   ```
   make --jobs=$(nproc --all) openvino_intel_cpu_plugin
   ```
   
## How to generate Conformance IRs set
Run the following commands:
1. Clone [`Open Model Zoo repo`](https://github.com/openvinotoolkit/open_model_zoo)
2. Download all possible models using [Downloader tool](https://github.com/openvinotoolkit/open_model_zoo/blob/master/tools/model_tools/downloader.py) from the repo.
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


# [Simple conformance runner](./../../../../ie_test_utils/functional_test_utils/layer_tests_summary/run_conformance.py)
There is simple python runner to complete whole conformance pipeline locally. Some steps could be excluded from the pipeline by command-line parameter configuration.

## The conformance pipeline steps:
1. (Optional) Download OMZ models
2. (Optional) Convert models to IR or prepare folder with IRs
3. (Optinal) Run `SubgraphDumper` to generate simple single op graph based on models or download `conformance_ir` folder
4. run conformance test executable files
5. generate conformance reports 

## Command line arguments
The script has the following srguments:
* `-h, --help`          show this help message and exit
* `-m MODELS_PATH, --models_path MODELS_PATH`
                        Path to directory/ies contains models to dump subgraph (default way is download OMZ). If `--s=0` specify Conformance IRs directory
* `-d DEVICE, --device DEVICE`
                        Specify target device. Default value is CPU
* `-ov OV_PATH, --ov_path OV_PATH`
                        OV binary files path. The default way is try to find installed OV by `INTEL_OPENVINO_DIR` in environmet variables or to find the absolute path of OV repo (by using script path)
* `-w WORKING_DIR, --working_dir WORKING_DIR`
                        Specify working directory to save all artifacts as reports, model, conformance_irs and etc.
* `-t TYPE, --type TYPE`
                        Specify conformance type: `OP` or `API`. Default value is `OP`
* `-s DUMP_CONFORMANCE, --dump_conformance DUMP_CONFORMANCE`
                        Set '1' if you want to create Conformance IRs from custom models/Downloaded models. In other case set `0`. Default value is '1'

> **NOTE**
> All arguments are optional and have default values to reproduce OMZ conformance results.

## Examples of usage:
1. Use default way to reproduce opset conformance results on GPU:
```
python3 run_conformance.py -d GPU
``` 
2. Use conformance pipeline to check new models support on CPU plugin and save results to custom directory:
```
python3 run_conformance.py -m /path/to/new/model_irs -s=1 -w /path/to/working/dir -d CPU
``` 
3. Use custom OV build to check GNA conformance using pre-generated conformance_irs from share:
```
python3 run_conformance.py -m /path/to/conformance_irs -s=0 -ov /path/to/ov_repo_on_custom_branch -d GNA
``` 


> **IMPORTANT NOTE:**
> If you need to debug some conformance tests use run of binary as a default way. If you want to get conformance results or reproduce CI behavior use the simple python runner.

