# Hello Query Device C++ Sample {#openvino_inference_engine_samples_hello_query_device_README}

This sample demonstrates how to execute an query Inference Engine devices, prints their metrics and default configuration values, using [Query Device API feature](../../../docs/IE_DG/InferenceEngine_QueryAPI.md).

Hello Query Device C++ sample application demonstrates how to use the following Inference Engine C++ API in applications:

| Feature    | API  | Description |
|:---     |:--- |:---
|Available Devices|`InferenceEngine::Core::GetAvailableDevices`, `InferenceEngine::Core::GetMetric`, `InferenceEngine::Core::GetConfig`| Get available devices information and configuration for inference

Basic Inference Engine API is covered by [Hello Classification C++ sample](../hello_classification/README.md).

| Options  | Values |
|:---                              |:---
| Supported devices                | [All](../../../docs/IE_DG/supported_plugins/Supported_Devices.md) |
| Other language realization       | [Python](../../ie_bridges/python/sample/hello_query_device/README.md) |

## How It Works

The sample queries all available Inference Engine devices, prints their supported metrics and plugin configuration parameters.

## Building

To build the sample, please use instructions available at [Build the Sample Applications](../../../docs/IE_DG/Samples_Overview.md) section in Inference Engine Samples guide.

## Running

To see quired information, run the following:

```
<path_to_sample>/hello_query_device -h
Usage : hello_query_device
```

## Sample Output

The application prints all available devices with their supported metrics and default values for configuration parameters:

```
Available devices:
        Device: CPU
        Metrics:
                SUPPORTED_METRICS : [ AVAILABLE_DEVICES SUPPORTED_METRICS FULL_DEVICE_NAME OPTIMIZATION_CAPABILITIES SUPPORTED_CONFIG_KEYS RANGE_FOR_ASYNC_INFER_REQUESTS RANGE_FOR_STREAMS ]
                FULL_DEVICE_NAME : Intel(R) Core(TM) i5-8350U CPU @ 1.70GHz
                OPTIMIZATION_CAPABILITIES : [ FP32 FP16 INT8 BIN ]
                SUPPORTED_CONFIG_KEYS : [ CPU_BIND_THREAD CPU_THREADS_NUM CPU_THROUGHPUT_STREAMS DUMP_EXEC_GRAPH_AS_DOT DYN_BATCH_ENABLED DYN_BATCH_LIMIT ENFORCE_BF16 EXCLUSIVE_ASYNC_REQUESTS PERF_COUNT ]
                RANGE_FOR_ASYNC_INFER_REQUESTS : { 1, 1, 1 }
                RANGE_FOR_STREAMS : { 1, 8 }
        Default values for device configuration keys:
                CPU_BIND_THREAD : NUMA
                CPU_THREADS_NUM : 0
                CPU_THROUGHPUT_STREAMS : 1
                DUMP_EXEC_GRAPH_AS_DOT : ""
                DYN_BATCH_ENABLED : NO
                DYN_BATCH_LIMIT : 0
                ENFORCE_BF16 : NO
                EXCLUSIVE_ASYNC_REQUESTS : NO
                PERF_COUNT : NO

        Device: GPU
        Metrics:
                SUPPORTED_METRICS : [ AVAILABLE_DEVICES SUPPORTED_METRICS FULL_DEVICE_NAME OPTIMIZATION_CAPABILITIES SUPPORTED_CONFIG_KEYS RANGE_FOR_ASYNC_INFER_REQUESTS RANGE_FOR_STREAMS ]
                FULL_DEVICE_NAME : Intel(R) UHD Graphics 620 (iGPU)
                OPTIMIZATION_CAPABILITIES : [ FP32 BIN FP16 ]
                SUPPORTED_CONFIG_KEYS : [ CACHE_DIR CLDNN_ENABLE_FP16_FOR_QUANTIZED_MODELS CLDNN_GRAPH_DUMPS_DIR GPU_MAX_NUM_THREADS CLDNN_MEM_POOL CLDNN_NV12_TWO_INPUTS CLDNN_PLUGIN_PRIORITY CLDNN_PLUGIN_THROTTLE CLDNN_SOURCES_DUMPS_DIR GPU_ENABLE_LOOP_UNROLLING CONFIG_FILE DEVICE_ID DUMP_KERNELS DYN_BATCH_ENABLED EXCLUSIVE_ASYNC_REQUESTS GPU_THROUGHPUT_STREAMS PERF_COUNT TUNING_FILE TUNING_MODE ]
                RANGE_FOR_ASYNC_INFER_REQUESTS : { 1, 2, 1 }
                RANGE_FOR_STREAMS : { 1, 2 }
        Default values for device configuration keys:
                CACHE_DIR : ""
                CLDNN_ENABLE_FP16_FOR_QUANTIZED_MODELS : YES
                CLDNN_GRAPH_DUMPS_DIR : ""
                CLDNN_MEM_POOL : YES
                CLDNN_NV12_TWO_INPUTS : NO
                CLDNN_PLUGIN_PRIORITY : 0
                CLDNN_PLUGIN_THROTTLE : 0
                CLDNN_SOURCES_DUMPS_DIR : ""
                GPU_MAX_NUM_THREADS : 8
                GPU_ENABLE_LOOP_UNROLLING : YES
                CONFIG_FILE : ""
                DEVICE_ID : ""
                DUMP_KERNELS : NO
                DYN_BATCH_ENABLED : NO
                EXCLUSIVE_ASYNC_REQUESTS : NO
                GPU_THROUGHPUT_STREAMS : 1
                PERF_COUNT : NO
                TUNING_FILE : ""
                TUNING_MODE : TUNING_DISABLED
```

## See Also

- [Integrate the Inference Engine with Your Application](../../../docs/IE_DG/Integrate_with_customer_application_new_API.md)
- [Using Inference Engine Samples](../../../docs/IE_DG/Samples_Overview.md)
