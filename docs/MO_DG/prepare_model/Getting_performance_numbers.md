# Getting Performance Numbers {#openvino_docs_MO_DG_Getting_Performance_Numbers}


## Tip 1. Measure the Proper Set of Operations 

When evaluating performance of your model with the Inference Engine, you must measure the proper set of operations. To do so, consider the following tips: 

- Avoid including one-time costs like model loading.

- Track separately the operations that happen outside the Inference Engine, like video decoding. 

> **NOTE**: Some image pre-processing can be baked into the IR and accelerated. For more information, refer to [Model Optimizer Knobs Related to Performance](Additional_Optimizations.md)

## Tip 2. Getting Credible Performance Numbers 

You need to build your performance conclusions on reproducible data. Do the performance measurements with a large number of invocations of the same routine. Since the first iteration is almost always significantly slower than the subsequent ones, you can use an aggregated value for the execution time for final projections:

-	If the warm-up run does not help or execution time still varies, you can try running a large number of iterations and then average or find a mean of the results.
-	 For time values that range too much, use geomean.

Refer to the [Inference Engine Samples](../../OV_Runtime_UG/Samples_Overview.md) for code examples for the performance measurements. Almost every sample, except interactive demos, has a `-ni` option to specify the number of iterations.

## Getting performance numbers using OpenVINO tool 

To get performance numbers use our Benchmark app.  

[Benchmark App](../../../samples/cpp/benchmark_app/README.md) sample is the best performance reference.
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

For example, for the CPU throughput mode from the previous section, you can play with number of streams (`-nstreams` command-line param). 
Try different values of the `-nstreams` argument from `1` to a number of CPU cores and find one that provides the best performance. For example, on a 8-core CPU, compare the `-nstreams 1` (which is a latency-oriented scenario) to the `2`, `4` and `8` streams. Notice that `benchmark_app` automatically queries/creates/runs number of requests required to saturate the given number of streams. 

Finally, notice that when you don't specify number of streams with `-nstreams`, "AUTO" value for the streams is used, e.g. for the CPU this is [CPU_THROUGHPUT_AUTO](../../OV_Runtime_UG/supported_plugins/CPU.md). You can spot the actual value behind "AUTO" for your machine in the application output.
Notice that the "AUTO" number is not necessarily most optimal, so it is generally recommended to play either with the benchmark_app's "-nstreams" as described above, or via  [new Workbench tool](@ref workbench_docs_Workbench_DG_Introduction).This allows you to simplify the app-logic, as you don't need to combine multiple inputs into a batch to achieve good CPU performance.
Instead, it is possible to keep a separate infer request per camera or another source of input and process the requests in parallel using Async API.

## Comparing Performance with Native/Framework Code 

When comparing the Inference Engine performance with the framework or another reference code, make sure that both versions are as similar as possible:

-	Wrap exactly the inference execution (refer to the [Inference Engine Samples](../../OV_Runtime_UG/Samples_Overview.md) for examples).
-	Do not include model loading time.
-	Ensure the inputs are identical for the Inference Engine and the framework. For example, Caffe\* allows to auto-populate the input with random values. Notice that it might give different performance than on real images.
-	Similarly, for correct performance comparison, make sure the access pattern, for example, input layouts, is optimal for Inference Engine (currently, it is NCHW).
-	Any user-side pre-processing should be tracked separately.
-	Make sure to try the same environment settings that the framework developers recommend, for example, for TensorFlow*. In many cases, things that are more machine friendly, like respecting NUMA (see <a href="#cpu-checklist">CPU Checklist</a>), might work well for the Inference Engine as well.
-	If applicable, use batching with the Inference Engine.
-	If possible, demand the same accuracy. For example, TensorFlow allows `FP16` support, so when comparing to that, make sure to test the Inference Engine with the `FP16` as well.

## Using Tools <a name="using-tools"></a>

Whether you are tuning for the first time or doing advanced performance optimization, you need a a tool that provides accurate insights. Intel&reg; VTune&trade; Amplifier gives you the tool to mine it and interpret the profiling data.

Alternatively, you can gather the raw profiling data that samples report, the second chapter provides example of how to interpret these.

### Internal Inference Performance Counters <a name="performance-counters"></a>

Almost every sample (inspect command-line options for a specific sample with `-h`) supports a `-pc` command that outputs internal execution breakdown. Refer to the [samples code](../../OV_Runtime_UG/Samples_Overview.md) for the actual Inference Engine API behind that.

Below is example of CPU plugin output for a network (since the device is CPU, the layers wall clock `realTime` and the `cpu` time are the same):

```
conv1      EXECUTED       layerType: Convolution        realTime: 706        cpu: 706            execType: jit_avx2
conv2_1_x1  EXECUTED       layerType: Convolution        realTime: 137        cpu: 137            execType: jit_avx2_1x1
fc6        EXECUTED       layerType: Convolution        realTime: 233        cpu: 233            execType: jit_avx2_1x1
fc6_nChw8c_nchw      EXECUTED  layerType: Reorder           realTime: 20         cpu: 20             execType: reorder
out_fc6         EXECUTED       layerType: Output            realTime: 3          cpu: 3              execType: unknown
relu5_9_x2    OPTIMIZED_OUT     layerType: ReLU             realTime: 0          cpu: 0              execType: undef
```

This contains layers name (as seen in IR), layers type and execution statistics. Notice the `OPTIMIZED_OUT`, which indicates that the particular activation was fused into adjacent convolution. Also, the `unknown` stays for the Inference Engine specific CPU (helper) primitives that are not part of the Intel MKL-DNN.

Notice that there are some helper layers in the CPU execution breakdown, which were not presented in the original topology. These are automatically added by the plugin. For example, the `Reorder` re-packs the Intel MKL-DNN internal (blocked) layout to the regular plain NCHW (that the user expects as the output). As explained in the <a href="#device-specific-tips">Few Device-Specific Tips</a>, if your custom kernels introduces a lot of outstanding/expensive Reorders, consider blocked implementation for the kernels.

Notice that in the heterogeneous cases, there will be additional information on which subgraph the statistics is about (the first subgraph is GPU, so its `cpu`/host time is really small compared to the actual `realTime`):

```
subgraph1: squeeze1x1   	  EXECUTED       layerType: Convolution        realTime: 227    cpu:3    execType: GPU
…
subgraph2: detection_out      EXECUTED       layerType: DetectionOutput    realTime: 121 cpu:121  execType: unknown
…
```

As mentioned earlier, `unknown` here means CPU kernel with unknown (for example, not AVX2 or AVX512) acceleration path.
Since FPGA execution does not separate individual kernels, only bulk execution/data transfer statistics is available:

```
subgraph1: 1. input preprocessing (mean data/FPGA):EXECUTED   layerType: preprocessing   realTime: 129     cpu: 129
subgraph1: 2. input transfer to DDR:EXECUTED       layerType:                    realTime: 201        cpu: 0              
subgraph1: 3. FPGA execute time:EXECUTED           layerType:                    realTime: 3808       cpu: 0              subgraph1: 4. output transfer from DDR:EXECUTED    layerType:                    realTime: 55         cpu: 0              
subgraph1: 5. FPGA output postprocessing:EXECUTED  layerType:                    realTime: 7          cpu: 7              
subgraph1: 6. softmax/copy:   EXECUTED       layerType:                    realTime: 2          cpu: 2              
subgraph2: out_prob:          NOT_RUN        layerType: Output             realTime: 0          cpu: 0              
subgraph2: prob:              EXECUTED       layerType: SoftMax            realTime: 10         cpu: 10             
Total time: 4212     microseconds
```

The `softmax/copy` is a glue layer that connects the FPGA subgraph to the CPU subgraph (and copies the data).

### Intel&reg; VTune&trade; Examples <a name="vtune-examples"></a>

All major performance calls of the Inference Engine are instrumented with Instrumentation and Tracing Technology APIs. This allows viewing the Inference Engine calls on the Intel&reg; VTune&trade; timelines and aggregations plus correlating them to the underlying APIs, like OpenCL.  In turn, this enables careful per-layer execution breakdown.

When choosing the Analysis type in Intel&reg; VTune&trade; Amplifier, make sure to select the **Analyze user tasks, events, and counters** option:

![](vtune_option.png)

See the [corresponding section in the Intel® VTune™ Amplifier User's Guide](https://software.intel.com/en-us/vtune-amplifier-help-task-analysis) for details.

Example of Inference Engine calls:

-	On the Intel VTune Amplifier timeline.
	Notice that `Task_runNOThrow` is an Async API wrapper and it is executed in a different thread and triggers the Intel MKL-DNN execution:

	![](vtune_timeline.png)
	
-	In the Intel VTune Amplifier **Top-down view**, grouped by the **Task Domain**.
	Notice the `Task_runNoThrow` and `MKLDNN _INFER` that are bracketing the actual Intel MKL-DNN kernels execution:
	
	![](vtune_topdown_view.jpg)
	
Similarly, you can use any GPU analysis in the Intel VTune Amplifier and get general correlation with Inference Engine API as well as the execution breakdown for OpenCL kernels.

Just like with regular native application, further drill down in the counters is possible, however, this is mostly useful for <a href="#optimizing-custom-kernels">optimizing custom kernels</a>. Finally, with the Intel VTune Amplifier, the profiling is not limited to your user-level code (see the [corresponding section in the Intel&reg; VTune&trade; Amplifier User's Guide](https://software.intel.com/en-us/vtune-amplifier-help-analyze-performance)).
