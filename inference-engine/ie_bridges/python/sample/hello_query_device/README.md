# Hello Query Device Python* Sample

This topic demonstrates how to run the Hello Query Device sample application, which queries Inference Engine 
devices and prints their metrics and default configuration values. The sample shows 
how to use Query Device API feature. 


## How It Works

The sample queries all available Inference Engine devices and prints their supported metrics and plugin configuration parameters.
 

## Running

The sample has no command-line parameters. To see the report, run the following command:

```
python3 hello_query_device.py
```

## Sample Output

The application prints all available devices with their supported metrics and default values for configuration parameters. For example:

```
Available devices:
	Device: CPU
	Metrics:
		AVAILABLE_DEVICES: 0
		SUPPORTED_METRICS: AVAILABLE_DEVICES, SUPPORTED_METRICS, FULL_DEVICE_NAME, OPTIMIZATION_CAPABILITIES, SUPPORTED_CONFIG_KEYS, RANGE_FOR_ASYNC_INFER_REQUESTS, RANGE_FOR_STREAMS
		FULL_DEVICE_NAME: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz
		OPTIMIZATION_CAPABILITIES: WINOGRAD, FP32, INT8, BIN
		SUPPORTED_CONFIG_KEYS: CPU_BIND_THREAD, CPU_THREADS_NUM, CPU_THROUGHPUT_STREAMS, DUMP_EXEC_GRAPH_AS_DOT, DYN_BATCH_ENABLED, DYN_BATCH_LIMIT, EXCLUSIVE_ASYNC_REQUESTS, PERF_COUNT, RANGE_FOR_ASYNC_INFER_REQUESTS, RANGE_FOR_STREAMS
		RANGE_FOR_ASYNC_INFER_REQUESTS: 0, 6, 1
		RANGE_FOR_STREAMS: 1, 12

	Default values for device configuration keys:
		CPU_BIND_THREAD: YES
		CPU_THREADS_NUM: 0
		CPU_THROUGHPUT_STREAMS: 1
		DUMP_EXEC_GRAPH_AS_DOT: 
		DYN_BATCH_ENABLED: NO
		DYN_BATCH_LIMIT: 0
		EXCLUSIVE_ASYNC_REQUESTS: NO
		PERF_COUNT: NO
		RANGE_FOR_ASYNC_INFER_REQUESTS: 1
		RANGE_FOR_STREAMS: 6
```
## See Also
* [Using Inference Engine Samples](./docs/IE_DG/Samples_Overview.md)
