# Hello Query Device Python* Sample {#openvino_inference_engine_ie_bridges_python_sample_hello_query_device_README}

This sample demonstrates how to show Inference Engine devices and prints their metrics and default configuration values using [Query Device API feature](../../../../../docs/IE_DG/InferenceEngine_QueryAPI.md).

The following Inference Engine Python API is used in the application:

| Feature      | API                                      | Description           |
| :----------- | :--------------------------------------- | :-------------------- |
| Basic        | [IECore]                                 | Common API            |
| Query Device | [IECore.get_metric], [IECore.get_config] | Get device properties |

| Options                    | Values                                                                  |
| :------------------------- | :---------------------------------------------------------------------- |
| Supported devices          | [All](../../../../../docs/IE_DG/supported_plugins/Supported_Devices.md) |
| Other language realization | [C++](../../../../samples/hello_query_device/README.md)                           |

## How It Works

The sample queries all available Inference Engine devices and prints their supported metrics and plugin configuration parameters.

## Running

The sample has no command-line parameters. To see the report, run the following command:

```
python <path_to_sample>/hello_query_device.py
```

## Sample Output

The application prints all available devices with their supported metrics and default values for configuration parameters. (Some lines are not shown due to length.) For example:

```
[ INFO ] Creating Inference Engine
[ INFO ] Available devices:
[ INFO ] CPU :
[ INFO ]        SUPPORTED_METRICS:
[ INFO ]                AVAILABLE_DEVICES:
[ INFO ]                FULL_DEVICE_NAME: Intel(R) Core(TM) i5-8350U CPU @ 1.70GHz
[ INFO ]                OPTIMIZATION_CAPABILITIES: FP32, FP16, INT8, BIN
[ INFO ]                RANGE_FOR_ASYNC_INFER_REQUESTS: 1, 1, 1
[ INFO ]                RANGE_FOR_STREAMS: 1, 8
[ INFO ]
[ INFO ]        SUPPORTED_CONFIG_KEYS (default values):
[ INFO ]                CPU_BIND_THREAD: NUMA
[ INFO ]                CPU_THREADS_NUM: 0
[ INFO ]                CPU_THROUGHPUT_STREAMS: 1
[ INFO ]                DUMP_EXEC_GRAPH_AS_DOT:
[ INFO ]                DYN_BATCH_ENABLED: NO
[ INFO ]                DYN_BATCH_LIMIT: 0
[ INFO ]                ENFORCE_BF16: NO
[ INFO ]                EXCLUSIVE_ASYNC_REQUESTS: NO
[ INFO ]                PERF_COUNT: NO
[ INFO ]
[ INFO ] GNA :
[ INFO ]        SUPPORTED_METRICS:
[ INFO ]                AVAILABLE_DEVICES: GNA_SW
[ INFO ]                OPTIMAL_NUMBER_OF_INFER_REQUESTS: 1
[ INFO ]                FULL_DEVICE_NAME: GNA_SW
[ INFO ]                GNA_LIBRARY_FULL_VERSION: 2.0.0.1047
[ INFO ]
[ INFO ]        SUPPORTED_CONFIG_KEYS (default values):
[ INFO ]                EXCLUSIVE_ASYNC_REQUESTS: NO
[ INFO ]                GNA_COMPACT_MODE: NO
[ INFO ]                GNA_DEVICE_MODE: GNA_SW_EXACT
[ INFO ]                GNA_FIRMWARE_MODEL_IMAGE:
[ INFO ]                GNA_FIRMWARE_MODEL_IMAGE_GENERATION:
[ INFO ]                GNA_LIB_N_THREADS: 1
[ INFO ]                GNA_PRECISION: I16
[ INFO ]                GNA_PWL_UNIFORM_DESIGN: NO
[ INFO ]                GNA_SCALE_FACTOR: 1.000000
[ INFO ]                GNA_SCALE_FACTOR_0: 1.000000
[ INFO ]                PERF_COUNT: NO
[ INFO ]                SINGLE_THREAD: YES
[ INFO ]
[ INFO ] GPU :
[ INFO ]        SUPPORTED_METRICS:
[ INFO ]                AVAILABLE_DEVICES: 0
[ INFO ]                FULL_DEVICE_NAME: Intel(R) UHD Graphics 620 (iGPU)
[ INFO ]                OPTIMIZATION_CAPABILITIES: FP32, BIN, FP16
[ INFO ]                RANGE_FOR_ASYNC_INFER_REQUESTS: 1, 2, 1
[ INFO ]                RANGE_FOR_STREAMS: 1, 2
[ INFO ]
[ INFO ]        SUPPORTED_CONFIG_KEYS (default values):
[ INFO ]                CACHE_DIR:
[ INFO ]                CLDNN_ENABLE_FP16_FOR_QUANTIZED_MODELS: YES
[ INFO ]                CLDNN_GRAPH_DUMPS_DIR:
[ INFO ]                CLDNN_MEM_POOL: YES
[ INFO ]                CLDNN_NV12_TWO_INPUTS: NO
[ INFO ]                CLDNN_PLUGIN_PRIORITY: 0
[ INFO ]                CLDNN_PLUGIN_THROTTLE: 0
[ INFO ]                CLDNN_SOURCES_DUMPS_DIR:
[ INFO ]                CONFIG_FILE:
[ INFO ]                DEVICE_ID:
[ INFO ]                DUMP_KERNELS: NO
[ INFO ]                DYN_BATCH_ENABLED: NO
[ INFO ]                EXCLUSIVE_ASYNC_REQUESTS: NO
[ INFO ]                GPU_THROUGHPUT_STREAMS: 1
[ INFO ]                PERF_COUNT: NO
[ INFO ]                TUNING_FILE:
[ INFO ]                TUNING_MODE: TUNING_DISABLED
[ INFO ]
```
## See Also

- [Using Inference Engine Samples](../../../../../docs/IE_DG/Samples_Overview.md)

[IECore]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IECore.html
[IECore.get_metric]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IECore.html#af1cdf2ecbea6399c556957c2c2fdf8eb
[IECore.get_config]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IECore.html#a48764dec7c235d2374af8b8ef53c6363
