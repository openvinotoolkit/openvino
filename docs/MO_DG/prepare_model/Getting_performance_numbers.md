# Getting Performance Numbers {#openvino_docs_MO_DG_Getting_Performance_Numbers}


## Tip 1. Measure the Proper Set of Operations 

When evaluating performance of your model with the OpenVINO Runtime, you must measure the proper set of operations. To do so, consider the following tips: 

- Avoid including one-time costs like model loading.

- Track separately the operations that happen outside the OpenVINO Runtime, like video decoding. 

> **NOTE**: Some image pre-processing can be baked into the IR and accelerated accordingly. For more information, refer to [Embedding the Preprocessing](Additional_Optimizations.md). Also consider [_runtime_ preprocessing optimizations](../../optimization_guide/dldt_deployment_optimization_common).

## Tip 2. Getting Credible Performance Numbers 

You need to build your performance conclusions on reproducible data. Do the performance measurements with a large number of invocations of the same routine. Since the first iteration is almost always significantly slower than the subsequent ones, you can use an aggregated value for the execution time for final projections:

-	If the warm-up run does not help or execution time still varies, you can try running a large number of iterations and then average or find a mean of the results.
-	For time values that range too much, consider geomean.
-   Beware of the throttling and other power oddities. A device can exist in one of several different power states. When optimizing your model, for better performance data reproducibility consider fixing the device frequency. However the end to end (application) benchmarking should be also performed under real operational conditions.

## Tip 3. Measure Reference Performance Numbers with OpenVINO's benchmark_app 

To get performance numbers, use the dedicated [Benchmark App](../../../samples/cpp/benchmark_app/README.md) sample which is the best way to produce the performance reference.
It has a lot of device-specific knobs, but the primary usage is as simple as: 
```bash
$ ./benchmark_app –d GPU –m <model> -i <input>
```
to measure the performance of the model on the GPU. 
Or
```bash
$ ./benchmark_app –d CPU –m <model> -i <input>
```
to execute on the CPU instead.

Each of the [OpenVINO supported devices](../../OV_Runtime_UG/supported_plugins/Supported_Devices.md) offers performance settings that have command-line equivalents in the [Benchmark App](../../../samples/cpp/benchmark_app/README.md).
While these settings provide really low-level control and allow to leverage the optimal model performance on the _specific_ device, we suggest always starting the performance evaluation with the [OpenVINO High-Level Performance Hints](../../OV_Runtime_UG/performance_hints.md) first:
 - benchmark_app **-hint tput** -d 'device' -m 'path to your model'
 - benchmark_app **-hint latency** -d 'device' -m 'path to your model'

## Comparing Performance with Native/Framework Code 

When comparing the OpenVINO Runtime performance with the framework or another reference code, make sure that both versions are as similar as possible:

-	Wrap exactly the inference execution (refer to the  [Benchmark App](../../../samples/cpp/benchmark_app/README.md) for examples).
-	Do not include model loading time.
-	Ensure the inputs are identical for the OpenVINO Runtime and the framework. For example, beware of random values that can be used to populate the inputs.
-	Consider [Image Pre-processing and Conversion](../../OV_Runtime_UG/preprocessing_overview.md), while any user-side pre-processing should be tracked separately.
-   When applicable, leverage the [Dynamic Shapes support](../../OV_Runtime_UG/ov_dynamic_shapes.md)
-	If possible, demand the same accuracy. For example, TensorFlow allows `FP16` execution, so when comparing to that, make sure to test the OpenVINO Runtime with the `FP16` as well.

## Internal Inference Performance Counters and Execution Graphs <a name="performance-counters"></a>
Further, finer-grained insights into inference performance breakdown can be achieved with device-specific performance counters and/or execution graphs.
Both [C++](../../../samples/cpp/benchmark_app/README.md) and [Python](../../../tools/benchmark_tool/README.md) versions of the `benchmark_app` supports a `-pc` command-line parameter that outputs internal execution breakdown.

Below is example of CPU plugin output for a network (since the device is CPU, the layers wall clock `realTime` and the `cpu` time are the same):

```
conv1      EXECUTED       layerType: Convolution        realTime: 706        cpu: 706            execType: jit_avx2
conv2_1_x1  EXECUTED       layerType: Convolution        realTime: 137        cpu: 137            execType: jit_avx2_1x1
fc6        EXECUTED       layerType: Convolution        realTime: 233        cpu: 233            execType: jit_avx2_1x1
fc6_nChw8c_nchw      EXECUTED  layerType: Reorder           realTime: 20         cpu: 20             execType: reorder
out_fc6         EXECUTED       layerType: Output            realTime: 3          cpu: 3              execType: unknown
relu5_9_x2    OPTIMIZED_OUT     layerType: ReLU             realTime: 0          cpu: 0              execType: undef
```
This contains layers name (as seen in IR), layers type and execution statistics. Notice the `OPTIMIZED_OUT`, which indicates that the particular activation was fused into adjacent convolution.
Both benchmark_app versions also support "exec_graph_path" command-line option governing the OpenVINO to output the same per-layer execution statistics, but in the form of the plugin-specific [Netron-viewable](https://netron.app/) graph to the specified file.

Notice that on some devices, the execution graphs/counters may be pretty intrusive overhead-wise. 
Also, especially when performance-debugging the [latency case](../../optimization_guide/dldt_deployment_optimization_latency.md) notice that  the counters do not reflect the time spent in the plugin/device/driver/etc queues. If the sum of the counters is too different from the latency of an inference request, consider testing with less inference requests. For example running single [OpenVINO stream](../../optimization_guide/dldt_deployment_optimization_tput.md) with multiple requests would produce nearly identical counters as running single inference request, yet the actual latency can be quite different.

Finally, the performance statistics with both performance counters and execution graphs is averaged, so such a data for the [dynamically-shaped inputs](../../OV_Runtime_UG/ov_dynamic_shapes.md) should be measured carefully (ideally by isolating the specific shape and executing multiple times in a loop, to gather the reliable data).

OpenVINO in general and individual plugins are heavily instrumented with Intel® instrumentation and tracing technology (ITT), so another option is to compile the OpenVINO from the source code with the ITT enabled and using tools like [Intel® VTune™ Profiler](https://software.intel.com/en-us/vtune) to get detailed inference performance breakdown and additional insights in the application-level performance on the timeline view.