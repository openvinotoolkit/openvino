# Default Model Optimizer Optimizations {#openvino_docs_MO_DG_Default_Model_Optimizer_Optimizations}

As described in the [Model Optimizer Guide](../MO_DG/prepare_model/Prepare_Trained_Model.md), there are a number of device-agnostic optimizations the tool performs.  For example, certain primitives like linear operations (BatchNorm and ScaleShift), are automatically fused into convolutions. Generally, these layers should not be manifested in the resulting IR:

![](../img/resnet_269.png)

The picture above shows Caffe\* Resnet269\* topology. The left model is the original model, and the one on the right (after conversion) is the resulting model that the Model Optimizer produces, with BatchNorm and ScaleShift layers  fused into the convolution weights rather than constituting separate layers.

If you still see these operations, inspect the Model Optimizer output carefully while searching for warnings, such as on the tool being unable to fuse. For example, non-linear operations (like activations) in between convolutions and linear operations might prevent the fusing. If performance is of concern, try to change (and potentially re-train) the topology. Refer to the [Model Optimizer Guide](../MO_DG/prepare_model/Model_Optimization_Techniques.md) for more optimizations.

Notice that the activation (`_relu`) is not touched by the Model Optimizer, and while it can be merged into convolution as well, this is rather a device-specific optimization, covered by Inference Engine during the model loading time. You are encouraged to inspect performance counters from plugins that should indicate that these particular layers are not executed (“Optimized out”). For more information, refer to <a href="#performance-counters">Internal Inference Performance Counters</a>.

Getting performance numbers (tuning optimization for performance) 

## Tip 1: Measure the Proper Set of Operations 

When evaluating performance of your model with the Inference Engine, you must measure the proper set of operations. To do so, consider the following tips: 

Avoid including one-time costs like model loading. For examples 

Track separately the operations that happen outside the Inference Engine, like video decoding. 

>**NOTE**: Some image pre-processing can be baked into the IR and accelerated. For more information, refer to Model Optimizer Knobs Related to Performance. 


## Tip 2. Getting Credible Performance Numbers 

You need to build your performance conclusions on reproducible data. Do the performance measurements with a large number of invocations of the same routine. Since the first iteration is almost always significantly slower than the subsequent ones, you can use an aggregated value for the execution time for final projections:

-	If the warm-up run does not help or execution time still varies, you can try running a large number of iterations and then average or find a mean of the results.
-	 For time values that range too much, use geomean.

Refer to the [Inference Engine Samples](../IE_DG/Samples_Overview.md) for code examples for the performance measurements. Almost every sample, except interactive demos, has a `-ni` option to specify the number of iterations.

## Getting performance numbers using OpenVINO tool 

To get performance numbers use our Benchmark app.  

[Benchmark App](../../inference-engine/samples/benchmark_app/README.md) sample is the best performance reference.
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

Finally, notice that when you don't specify number of streams with `-nstreams`, "AUTO" value for the streams is used, e.g. for the CPU this is [CPU_THROUGHPUT_AUTO](supported_plugins/CPU.md). You can spot the actual value behind "AUTO" for your machine in the application output.
Notice that the "AUTO" number is not necessarily most optimal, so it is generally recommended to play either with the benchmark_app's "-nstreams" as described above, or via  [new Workbench tool](@ref workbench_docs_Workbench_DG_Introduction).This allows you to simplify the app-logic, as you don't need to combine multiple inputs into a batch to achieve good CPU performance.
Instead, it is possible to keep a separate infer request per camera or another source of input and process the requests in parallel using Async API.

## Comparing Performance with Native/Framework Code 

When comparing the Inference Engine performance with the framework or another reference code, make sure that both versions are as similar as possible:

-	Wrap exactly the inference execution (refer to the [Inference Engine Samples](../IE_DG/Samples_Overview.md) for examples).
-	Do not include model loading time.
-	Ensure the inputs are identical for the Inference Engine and the framework. For example, Caffe\* allows to auto-populate the input with random values. Notice that it might give different performance than on real images.
-	Similarly, for correct performance comparison, make sure the access pattern, for example, input layouts, is optimal for Inference Engine (currently, it is NCHW).
-	Any user-side pre-processing should be tracked separately.
-	Make sure to try the same environment settings that the framework developers recommend, for example, for TensorFlow*. In many cases, things that are more machine friendly, like respecting NUMA (see <a href="#cpu-checklist">CPU Checklist</a>), might work well for the Inference Engine as well.
-	If applicable, use batching with the Inference Engine.
-	If possible, demand the same accuracy. For example, TensorFlow allows `FP16` support, so when comparing to that, make sure to test the Inference Engine with the `FP16` as well.