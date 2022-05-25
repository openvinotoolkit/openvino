# Getting Performance Numbers {#openvino_docs_MO_DG_Getting_Performance_Numbers}

This guide introduces things to notice and how to use the benchmark_app to get performance numbers. It also explains how the performance numbers are reflected through internal inference performance counters and execution graphs. In the last section, it includes information on using ITT and Intel® VTune™ Profiler to get performance insights.

## Tip 1: Select Proper Set of Operations to Measure

When evaluating the performance of a model with OpenVINO Runtime, it is required to measure proper set of operations. Remember the following tips:
- Avoid including one-time costs such as model loading.

- Track operations that occur outside OpenVINO Runtime (such as video decoding) separately. 

> **NOTE**: Some image pre-processing can be baked into OpenVINO IR and accelerated accordingly. For more information, refer to [Embedding the Pre-processing](Additional_Optimizations.md) and [General Runtime Optimizations](../../optimization_guide/dldt_deployment_optimization_common).

## Tip 2: Try to Get Credible Data

Performance conclusions should be build upon reproducible data. As for the performance measurements, they should be done with a large number of invocations of the same routine. Since the first iteration is almost always significantly slower than the subsequent ones, an aggregated value can be used for the execution time for final projections:

-	If the warm-up run does not help or execution time still varies, you can try running a large number of iterations and then average or find a mean of the results.
-	If the time values range too much, consider geomean.
-  Be aware of the throttling and other power oddities. A device can exist in one of several different power states. When optimizing your model, consider fixing the device frequency for better performance data reproducibility. However, the end-to-end (application) benchmarking should also be performed under real operational conditions.

## Using benchmark_app to Measure Reference Performance Numbers 

To get performance numbers, use the dedicated [OpenVINO Benchmark app](../../../samples/cpp/benchmark_app/README.md) sample, which is the most-recommended solution to produce performance reference.
It includes a lot of device-specific knobs, but the primary usage is as simple as: 
```bash
$ ./benchmark_app –d GPU –m <model> -i <input>
```
to measure the performance of the model on the GPU. 
Or
```bash
$ ./benchmark_app –d CPU –m <model> -i <input>
```
to execute on the CPU instead.

Each of the [OpenVINO supported devices](../../OV_Runtime_UG/supported_plugins/Supported_Devices.md) offers performance settings that contain command-line equivalents in the [Benchmark App](../../../samples/cpp/benchmark_app/README.md).
While these settings provide really low-level control and allow leveraging the optimal model performance on the _specific_ device, it is suggested to always start the performance evaluation with the [OpenVINO High-Level Performance Hints](../../OV_Runtime_UG/performance_hints.md) first:
 - benchmark_app **-hint tput** -d 'device' -m 'path to your model'
 - benchmark_app **-hint latency** -d 'device' -m 'path to your model'

## Comparing Performance with Native/Framework Code 

When comparing the OpenVINO Runtime performance with the framework or another reference code, make sure that both versions are as similar as possible:

-	Wrap the inference execution exactly (for examples, refer to the [Benchmark App](../../../samples/cpp/benchmark_app/README.md)).
-	Do not include model loading time.
-	Ensure the inputs are identical for the OpenVINO Runtime and the framework. For example, watch out for random values that can be used to populate the inputs.
-	In situations when any user-side pre-processing should be tracked separately, consider [Image Pre-processing and Conversion](../../OV_Runtime_UG/preprocessing_overview.md).
-  When applicable, leverage the [Dynamic Shapes support](../../OV_Runtime_UG/ov_dynamic_shapes.md)
-	If possible, demand the same accuracy. For example, TensorFlow allows `FP16` execution, so when comparing to that, make sure to test the OpenVINO Runtime with the `FP16` as well.

## Internal Inference Performance Counters and Execution Graphs <a name="performance-counters"></a>
More detailed insights into inference performance breakdown can be achieved with device-specific performance counters and/or execution graphs.
Both [C++](../../../samples/cpp/benchmark_app/README.md) and [Python](../../../tools/benchmark_tool/README.md) versions of the `benchmark_app` support a `-pc` command-line parameter that outputs internal execution breakdown.

For example, below is the part of performance counters for quantized [TensorFlow implementation of ResNet-50](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/resnet-50-tf) model inference on [CPU Plugin](../../OV_Runtime_UG/supported_plugins/CPU.md).
Keep in mind that since the device is CPU, the `realTime` wall clock and the `cpu` time layers are the same. Information about layer precision is also stored in the performance counters. 

| layerName                                                 | execStatus | layerType    | execType             | realTime (ms) | cpuTime (ms) |
| --------------------------------------------------------- | ---------- | ------------ | -------------------- | ------------- | ------------ |
| resnet\_model/batch\_normalization\_15/FusedBatchNorm/Add | EXECUTED   | Convolution  | jit\_avx512\_1x1\_I8 | 0.377         | 0.377        |
| resnet\_model/conv2d\_16/Conv2D/fq\_input\_0              | NOT\_RUN   | FakeQuantize | undef                | 0             | 0            |
| resnet\_model/batch\_normalization\_16/FusedBatchNorm/Add | EXECUTED   | Convolution  | jit\_avx512\_I8      | 0.499         | 0.499        |
| resnet\_model/conv2d\_17/Conv2D/fq\_input\_0              | NOT\_RUN   | FakeQuantize | undef                | 0             | 0            |
| resnet\_model/batch\_normalization\_17/FusedBatchNorm/Add | EXECUTED   | Convolution  | jit\_avx512\_1x1\_I8 | 0.399         | 0.399        |
| resnet\_model/add\_4/fq\_input\_0                         | NOT\_RUN   | FakeQuantize | undef                | 0             | 0            |
| resnet\_model/add\_4                                      | NOT\_RUN   | Eltwise      | undef                | 0             | 0            |
| resnet\_model/add\_5/fq\_input\_1                         | NOT\_RUN   | FakeQuantize | undef                | 0             | 0            |


   The `exeStatus` column of the table includes possible values:
   - The `EXECUTED` - layer was executed by standalone primitive.
   - The `NOT_RUN` - layer was not executed by standalone primitive or was fused with another operation and executed in another layer primitive.  
   
   The `execType` column of the table includes inference primitives with specific suffixes. The layers have the following marks:
   * Suffix `I8` - layers that had 8-bit data type input and were computed in 8-bit precision.
   * Suffix `FP32` - layers computed in 32-bit precision.

   All `Convolution` layers are executed in `int8` precision. Rest layers are fused into Convolutions using post operations optimization technique (for more details, see the [Internal CPU Plugin Optimizations](../../OV_Runtime_UG/supported_plugins/CPU.md)).
   This contains name (as seen in IR) and type of the layer and execution statistics.

Both `benchmark_app` versions also support `exec_graph_path` command-line option. It orders the OpenVINO to output the same per-layer execution statistics, but in the form of the plugin-specific [Netron-viewable](https://netron.app/) graph to the specified file.

Keep in mind that on some devices, the execution graphs/counters may be pretty intrusive overhead-wise. 
Be aware, especially when performance-debugging the [latency case](../../optimization_guide/dldt_deployment_optimization_latency.md), that the counters do not reflect the time spent in the plugin/device/driver/etc queues. If the sum of the counters is too different from the latency of an inference request, consider testing with less inference requests. For example, running single [OpenVINO stream](../../optimization_guide/dldt_deployment_optimization_tput.md) with multiple requests would produce nearly identical counters as running single inference request, yet the actual latency can be quite different.

Finally, the performance statistics with both performance counters and execution graphs is averaged, so such a data for the [dynamically-shaped inputs](../../OV_Runtime_UG/ov_dynamic_shapes.md) should be measured carefully. Preferably by isolating the specific shape and executing multiple times in a loop, to gather the reliable data.

In general, OpenVINO and its individual plugins are heavily instrumented with Intel® instrumentation and tracing technology (ITT). Another option is to compile the OpenVINO from the source code with the ITT enabled and using tools like [Intel® VTune™ Profiler](https://software.intel.com/en-us/vtune) to get detailed inference performance breakdown and additional insights in the application-level performance on the timeline view.
