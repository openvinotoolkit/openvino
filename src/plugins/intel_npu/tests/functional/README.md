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
* `OV_NPU_TESTS_SKIP_CONFIG_FILE` - string type, path to skip filters file

## Skip filter 

Skip filters are used to select which tests can run on specific devices, backends or Operating Systems.
By default `ov_npu_func_tests` does not have any skips filters configured, to enable them it's necessary to set an environment variable with the path to the skip config file.

By default, the environment variable `OV_NPU_TESTS_SKIP_CONFIG_FILE` is set to find skip_tests.xml in the current working folder.
`OV_NPU_TESTS_SKIP_CONFIG_FILE` has to be set with a valid path to an .xml file containing filters with the following structure:

```xml
<skip_configs>
    <skip_config>
        <message>skip_message_xxxxxx</message>
        <enable_rules>
            <backend>LEVEL0</backend>
            <backend></backend> (empty brackets denote no backend)
            <device>3720</device>
            <device>!4000</device> (using "!" to negate rule)
            <driver_version>9999</driver_version>
            <operating_system>windows</operating_system>
            <operating_system>linux</operating_system>
        </enable_rules>
        <filters>
            <filter>skip_filter_xxxxxxxxxx</filter>
            <filter>skip_filter_xxxxxxxxxx</filter>
            <filter>skip_filter_xxxxxxxxxx</filter>
        </filters>
    </skip_config>
</skip_configs>
```

Skip filters can be enabled/disabled according to rules defining the device, backend, driver version or operating system, depending on where tests are supposed to run.
Rules are optional, multiple rules can be chained together. Users can negate a rule by using "!".
When determining if a skip filter is active, rules across different categories (backend, device, driver_version, operating_system) are combined using an AND operation. While multiple entries of the same category will be evaluated using the OR operation.

## Folder structure

### Behavior
The folder structure of the `behavior` directory follows the folder structure of the OpenVINO base tests.

The OpenVINO convention requires the tests to be organized in at least three directories: `compiled_model`, `ov_infer_request`, and `ov_plugin`. If the scope of NPU Plugin tests becomes broader, additional directories may be created.

The E2E tests reside here.

### Common
The place of utility functions and `getTestCaseName` base implementations.

### Internal
The folder structure is mirrored from the one in the `intel_npu/src` directory.

The NPU implementations of test classes and their corresponding tests are located here.

### Shared tests instances
OpenVINO test instantiations using NPU Plugin's own arguments.

## Test developing convention

Any test should be split into a `.hpp` and `.cpp` file, with the exception of the tests from `shared_tests_instances`, which don't need `.hpp` files. A test's `.hpp` file should be located in the same directory as its `.cpp` file.

The `.hpp` file should have the entirety of its contents wrapped around the `ov::test::behavior` namespace.
The contents are:
- class declarations
- methods/functions definitions
- test definitions
- aliases

The `.cpp` file contains the test instantiations and the variables passed as arguments. The contents should be wrapped around an anonymous/unnamed namespace.
