# Hello Query Device C++ Sample

This topic demonstrates how to run the Hello Query Device sample application, which queries Inference Engine devices and prints their metrics and default configuration values. The sample shows how to use [Query Device API feature](./docs/IE_DG/InferenceEngine_QueryAPI.md).
> **NOTE:** This topic describes usage of C++ implementation of the Query Device Sample. 
> For the Python* implementation, refer to [Hello Query Device Python* Sample](./inference-engine/ie_bridges/python/sample/hello_query_device/README.md)
## Running

To see quired information, run the following:
```sh
./hello_query_device
```

## Sample Output

The application prints all available devices with their supported metrics and default values for configuration parameters:

```
Available devices: 
	Device: CPU
	Metrics: 
		AVAILABLE_DEVICES : [ 0 ]
		SUPPORTED_METRICS : [ AVAILABLE_DEVICES SUPPORTED_METRICS FULL_DEVICE_NAME OPTIMIZATION_CAPABILITIES SUPPORTED_CONFIG_KEYS RANGE_FOR_ASYNC_INFER_REQUESTS RANGE_FOR_STREAMS ]
		FULL_DEVICE_NAME : Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz
		OPTIMIZATION_CAPABILITIES : [ WINOGRAD FP32 INT8 BIN ]
		SUPPORTED_CONFIG_KEYS : [ CPU_BIND_THREAD CPU_THREADS_NUM CPU_THROUGHPUT_STREAMS DUMP_EXEC_GRAPH_AS_DOT DYN_BATCH_ENABLED DYN_BATCH_LIMIT EXCLUSIVE_ASYNC_REQUESTS PERF_COUNT ]
		...
	Default values for device configuration keys: 
		CPU_BIND_THREAD : YES
		CPU_THREADS_NUM : 0
		CPU_THROUGHPUT_STREAMS : 1
		DUMP_EXEC_GRAPH_AS_DOT : ""
		DYN_BATCH_ENABLED : NO
		DYN_BATCH_LIMIT : 0
		EXCLUSIVE_ASYNC_REQUESTS : NO
		PERF_COUNT : NO

	Device: FPGA
	Metrics: 
		AVAILABLE_DEVICES : [ 0 ]
		SUPPORTED_METRICS : [ AVAILABLE_DEVICES SUPPORTED_METRICS SUPPORTED_CONFIG_KEYS FULL_DEVICE_NAME OPTIMIZATION_CAPABILITIES RANGE_FOR_ASYNC_INFER_REQUESTS ]
		SUPPORTED_CONFIG_KEYS : [ DEVICE_ID PERF_COUNT EXCLUSIVE_ASYNC_REQUESTS DLIA_IO_TRANSFORMATIONS_NATIVE DLIA_ARCH_ROOT_DIR DLIA_PERF_ESTIMATION ]
		FULL_DEVICE_NAME : a10gx_2ddr : Intel Vision Accelerator Design with Intel Arria 10 FPGA (acla10_1150_sg10)
		OPTIMIZATION_CAPABILITIES : [ FP16 ]
		RANGE_FOR_ASYNC_INFER_REQUESTS : { 2, 5, 1 }
	Default values for device configuration keys: 
		DEVICE_ID : [ 0 ]
		PERF_COUNT : true
		EXCLUSIVE_ASYNC_REQUESTS : false
		DLIA_IO_TRANSFORMATIONS_NATIVE : false
		DLIA_PERF_ESTIMATION : true
```

## See Also
* [Using Inference Engine Samples](./docs/IE_DG/Samples_Overview.md)
* [Model Downloader](https://github.com/opencv/open_model_zoo/tree/2018/model_downloader)
* [Model Optimizer](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
