# Introduction to the Performance Topics {#openvino_docs_IE_DG_Intro_to_Performance}

This section is a shorter version of the
[Optimization Guide](../optimization_guide/dldt_optimization_guide.md) for the Intel® Distribution of OpenVINO™ Toolkit.

## Precision
Inference precision directly affects the performance. 

Model Optimizer can produce an IR with different precision. For example, an FP16 IR initially targets VPU and GPU devices, while, for example, for the CPU, an FP16 IR is    typically up-scaled to the regular FP32 automatically upon loading. But notice that further device-specific inference precision settings are available, 
for example, [8-bit integer](Int8Inference.md) or [bfloat16](Bfloat16Inference.md), which is specific to the CPU inference, below.
Note that for the [MULTI device](supported_plugins/MULTI.md) plugin that supports automatic inference on multiple devices in parallel, you can use an FP16 IR (no need for FP32).
You can find more information, including preferred data types for specific devices, in the
[Supported Devices](supported_plugins/Supported_Devices.md) document.

## Automatic Lowering of the Inference Precision
By default, plugins enable the optimizations that allow lower precision if the acceptable range of accuracy is preserved.
For example, for the CPU that supports the AVX512_BF16 instructions, an FP16/FP32 model is converted to a [bfloat16](Bfloat16Inference.md) IR to accelerate inference.
To compare the associated speedup, run the example command below to disable this feature on the CPU device with the AVX512_BF16 support and get regular FP32 execution:
```
$ benchmark_app -m <model.xml> -enforcebf16=false
 ```
Notice that for quantized (e.g. INT8) models the bfloat16 calculations (of the layers that remain in FP32) is disabled by default.
Refer to the [CPU Plugin documentation](supported_plugins/CPU.md) for more details.

Similarly, the GPU device automatically executes FP16 for the layers that remain in FP16 in the quantized models (assuming that the FP16 model was quantized).
Refer to the ENABLE_FP16_FOR_QUANTIZED_MODELS key in the [GPU Plugin documentation](supported_plugins/GPU.md).

## Latency vs. Throughput
One way to increase computational efficiency is batching, which combines many (potentially tens) of
input images to achieve optimal throughput. However, high batch size also comes with a
latency penalty. So, for more real-time oriented usages, lower batch sizes (as low as a single input) are used.
Refer to the [Benchmark App](../../inference-engine/samples/benchmark_app/README.md) sample, which allows latency vs. throughput measuring.

## Using Caching API for first inference latency optimization
Since with the 2021.4 release, Inference Engine provides an ability to enable internal caching of loaded networks.
This can significantly reduce load network latency for some devices at application startup.
Internally caching uses plugin's Export/ImportNetwork flow, like it is done for [Compile tool](../../inference-engine/tools/compile_tool/README.md), using the regular ReadNetwork/LoadNetwork API.
Refer to the [Model Caching Overview](Model_caching_overview.md) for more detailed explanation.

## Using Async API
To gain better performance on accelerators, such as VPU, the Inference Engine uses the asynchronous approach (see
[Integrating Inference Engine in Your Application (current API)](Integrate_with_customer_application_new_API.md)).
The point is amortizing the costs of data transfers, by pipe-lining, see [Async API explained](@ref omz_demos_object_detection_demo_cpp).
Since the pipe-lining relies on the availability of the parallel slack, running multiple inference requests in parallel is essential.
Refer to the [Benchmark App](../../inference-engine/samples/benchmark_app/README.md) sample, which enables running a number of inference requests in parallel. Specifying different number of request produces different throughput measurements.

## Best Latency on the Multi-Socket CPUs
Note that when latency is of concern, there are additional tips for multi-socket systems.
When input is limited to the single image, the only way to achieve the best latency is to limit execution to the single socket.
The reason is that single image is simply not enough
to saturate more than one socket. Also NUMA overheads might dominate the execution time.
Below is the example command line that limits the execution to the single socket using numactl for the best *latency* value
(assuming the machine with 28 phys cores per socket):
```
limited to the single socket).
$ numactl -m 0 --physcpubind 0-27  benchmark_app -m <model.xml> -api sync -nthreads 28
 ```
Note that if you have more than one input, running as many inference streams as you have NUMA nodes (or sockets)
usually gives the same best latency as a single request on the single socket, but much higher throughput. Assuming two NUMA nodes machine:
```
$ benchmark_app -m <model.xml> -nstreams 2
 ```
Number of NUMA nodes on the machine can be queried via 'lscpu'.
Please see more on the NUMA support in the [Optimization Guide](../optimization_guide/dldt_optimization_guide.md).

## Throughput Mode for CPU
Unlike most accelerators, CPU is perceived as an inherently latency-oriented device. 
OpenVINO™ toolkit provides a "throughput" mode that allows running multiple inference requests on the CPU simultaneously, which greatly improves the throughput.

Internally, the execution resources are split/pinned into execution "streams".
Using this feature gains much better performance for the networks that originally are not scaled well with a number of threads (for example, lightweight topologies). This is especially pronounced for the many-core server machines.

Run the [Benchmark App](../../inference-engine/samples/benchmark_app/README.md) and play with number of infer requests running in parallel, next section. 
Try different values of the `-nstreams` argument from `1` to a number of CPU cores and find one that provides the best performance. 

The throughput mode relaxes the requirement to saturate the CPU by using a large batch: running multiple independent inference requests in parallel often gives much better performance, than using a batch only.
This allows you to simplify the app-logic, as you don't need to combine multiple inputs into a batch to achieve good CPU performance.
Instead, it is possible to keep a separate infer request per camera or another source of input and process the requests in parallel using Async API.

## Benchmark App
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
