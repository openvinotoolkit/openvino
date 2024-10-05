# OpenVINO Functional test suite

Test binary `ov_npu_func_tests` is built with GTest framework and includes test instances which are common
for other OpenVINO plugins as well as test instances specific for this plugin.
`ov_npu_func_tests --help` will show the usage.

## Environment variables

The following environment variables can be set up for the run of test binary `ov_npu_func_tests`. Values for boolean variables are specified as `1` or `0`.
* `IE_NPU_TESTS_DUMP_PATH` - string type, target directory for the exported artifacts and source directory for imported artifacts
* `IE_NPU_TESTS_DEVICE_NAME` - string type, passed to the Inference Engine as the name of device for loading the appropriate plugin. Default - `NPU`
* `IE_NPU_TESTS_LOG_LEVEL` - string type, passed to the plugin and allows additional debug output to the console. Sample value - `LOG_DEBUG`
* `IE_NPU_TESTS_RUN_EXPORT` - bool type, denotes whether to export produced networks as artifacts to the files matching current test case
* `IE_NPU_TESTS_RUN_IMPORT` - bool type, denotes whether to import the networks from the files matching current test case (instead of compiling them on the fly)
* `IE_NPU_TESTS_RUN_INFER` - bool type, denotes whether to execute infer request which as part of the test case or not
* `IE_NPU_TESTS_EXPORT_INPUT` - bool type, denotes whether to export produced input data as artifacts to the files matching current test case
* `IE_NPU_TESTS_EXPORT_REF` - bool type, denotes whether to export calculated reference values as artifacts to the files matching current test case
* `IE_NPU_TESTS_IMPORT_INPUT` - bool type, denotes whether to read input data from the files matching current test case (instead of using generated data)
* `IE_NPU_TESTS_IMPORT_REF` - bool type, denotes whether to read reference values from the files matching current test case (instead of calculating them)
* `IE_NPU_TESTS_RAW_EXPORT` - bool type, denotes whether to use header for exported network file or not
* `IE_NPU_TESTS_LONG_FILE_NAME` - bool type, denotes whether to allow longer file names for the exported artifacts. By default shorter file names are used for all operating systems
* `IE_NPU_TESTS_PLATFORM` - string type, enable compiler config option `NPU_PLATFORM` with value from the environment. Sample value - `NPU3720`. For more information about possible values, refer to the [NPU plugin README file](./../../../intel_npu/README.md).
