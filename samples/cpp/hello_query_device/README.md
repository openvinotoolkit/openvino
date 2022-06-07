# Hello Query Device C++ Sample {#openvino_inference_engine_samples_hello_query_device_README}

This sample demonstrates how to execute an query OpenVINO™ Runtime devices, prints their metrics and default configuration values, using [Properties API](../../../docs/OV_Runtime_UG/supported_plugins/config_properties.md). The source code for this example is also available [on GitHub](https://github.com/openvinotoolkit/openvino/tree/master/samples/cpp/hello_query_device).

The following C++ API is used in the application:

| Feature | API | Description |
| :--- | :--- | :--- |
| Available Devices | `ov::Core::get_available_devices`, `ov::Core::get_property` | Get available devices information and configuration for inference |

Basic OpenVINO™ Runtime API is covered by [Hello Classification C++ sample](../hello_classification/README.md).

| Options  | Values |
| :--- |:---
| Supported devices | [All](../../../docs/OV_Runtime_UG/supported_plugins/Supported_Devices.md) |
| Other language realization | [Python](../../python/hello_query_device/README.md) |

## How It Works

The sample queries all available OpenVINO™ Runtime devices, prints their supported metrics and plugin configuration parameters.

## Building

To build the sample, please use instructions available at [Build the Sample Applications](../../../docs/OV_Runtime_UG/Samples_Overview.md) section in OpenVINO™ Toolkit Samples guide.

## Running

To see quired information, run the following:

```
hello_query_device
```

## Sample Output

The application prints all available devices with their supported metrics and default values for configuration parameters:

```
[ INFO ] OpenVINO Runtime version ......... <version>
[ INFO ] Build ........... <build>
[ INFO ]
[ INFO ] Available devices:
[ INFO ] CPU
[ INFO ]        SUPPORTED_METRICS:
[ INFO ]                AVAILABLE_DEVICES : [  ]
[ INFO ]                FULL_DEVICE_NAME : Intel(R) Core(TM) i5-8350U CPU @ 1.70GHz
[ INFO ]                OPTIMIZATION_CAPABILITIES : [ FP32 FP16 INT8 BIN ]
[ INFO ]                RANGE_FOR_ASYNC_INFER_REQUESTS : { 1, 1, 1 }
[ INFO ]                RANGE_FOR_STREAMS : { 1, 8 }
[ INFO ]                IMPORT_EXPORT_SUPPORT : true
[ INFO ]        SUPPORTED_CONFIG_KEYS (default values):
[ INFO ]                CACHE_DIR : ""
[ INFO ]                CPU_BIND_THREAD : NO
[ INFO ]                CPU_THREADS_NUM : 0
[ INFO ]                CPU_THROUGHPUT_STREAMS : 1
[ INFO ]                DUMP_EXEC_GRAPH_AS_DOT : ""
[ INFO ]                DYN_BATCH_ENABLED : NO
[ INFO ]                DYN_BATCH_LIMIT : 0
[ INFO ]                ENFORCE_BF16 : NO
[ INFO ]                EXCLUSIVE_ASYNC_REQUESTS : NO
[ INFO ]                PERFORMANCE_HINT : ""
[ INFO ]                PERFORMANCE_HINT_NUM_REQUESTS : 0
[ INFO ]                PERF_COUNT : NO
[ INFO ]
[ INFO ] GNA
[ INFO ]        SUPPORTED_METRICS:
[ INFO ]                AVAILABLE_DEVICES : [ GNA_SW_EXACT ]
[ INFO ]                OPTIMAL_NUMBER_OF_INFER_REQUESTS : 1
[ INFO ]                FULL_DEVICE_NAME : GNA_SW_EXACT
[ INFO ]                GNA_LIBRARY_FULL_VERSION : 3.0.0.1455
[ INFO ]                IMPORT_EXPORT_SUPPORT : true
[ INFO ]        SUPPORTED_CONFIG_KEYS (default values):
[ INFO ]                EXCLUSIVE_ASYNC_REQUESTS : NO
[ INFO ]                GNA_COMPACT_MODE : YES
[ INFO ]                GNA_COMPILE_TARGET : ""
[ INFO ]                GNA_DEVICE_MODE : GNA_SW_EXACT
[ INFO ]                GNA_EXEC_TARGET : ""
[ INFO ]                GNA_FIRMWARE_MODEL_IMAGE : ""
[ INFO ]                GNA_FIRMWARE_MODEL_IMAGE_GENERATION : ""
[ INFO ]                GNA_LIB_N_THREADS : 1
[ INFO ]                GNA_PRECISION : I16
[ INFO ]                GNA_PWL_MAX_ERROR_PERCENT : 1.000000
[ INFO ]                GNA_PWL_UNIFORM_DESIGN : NO
[ INFO ]                GNA_SCALE_FACTOR : 1.000000
[ INFO ]                GNA_SCALE_FACTOR_0 : 1.000000
[ INFO ]                LOG_LEVEL : LOG_NONE
[ INFO ]                PERF_COUNT : NO
[ INFO ]                SINGLE_THREAD : YES
```

## See Also

- [Integrate the OpenVINO™ Runtime with Your Application](../../../docs/OV_Runtime_UG/integrate_with_your_application.md)
- [Using OpenVINO™ Toolkit Samples](../../../docs/OV_Runtime_UG/Samples_Overview.md)
