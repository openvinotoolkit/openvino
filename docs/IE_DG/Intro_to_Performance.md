# Introduction to the Performance Topics {#openvino_docs_IE_DG_Intro_to_Performance}

This section is a shorter version of the
[Optimization Guide](supported_plugins/MULTI.md) for the Intel Deep Learning Deployment Toolkit.

## Precision
Inference precision directly affects the performance. 

Model Optimizer can produce an IR with different precision. For example, float16 IR initially targets VPU and GPU devices, while, for example, the CPU can also execute regular float32.
Also, further device-specific inference precision settings are available, for example, [8-bit integer](Int8Inference.md) or [bfloat16](Bfloat16Inference.md) inference on the CPU.
Note that for [MULTI device](supported_plugins/MULTI.md) that supports automatic inference on multiple devices in parallel, you can use the FP16 IR.
You can find more information, including preferred data types for specific devices, in the
[Supported Devices](supported_plugins/Supported_Devices.md) section.

## Lowering Inference Precision
Default optimization is used for CPU and implies that inference is made with lower precision if it is possible on a given platform to reach better performance with acceptable range of accuracy.
This approach is used for CPU device if platform supports the AVX512_BF16 instruction. In this case, a regular float32 model is converted to [bfloat16](Bfloat16Inference.md) internal representation and inference is provided with bfloat16 layers usage.
Below is the example command line to disable this feature on the CPU device with the AVX512_BF16 instruction and execute regular float32.
```
$ benchmark_app -m <model.xml> -enforcebf16=false
 ```

## Latency vs. Throughput
One way to increase computational efficiency is batching, which combines many (potentially tens) of
input images to achieve optimal throughput. However, high batch size also comes with a
latency penalty. So, for more real-time oriented usages, lower batch sizes (as low as a single input) are used.
Refer to the [Benchmark App](../../inference-engine/samples/benchmark_app/README.md) sample, which allows latency vs. throughput measuring.

## Using Async API
To gain better performance on accelerators, such as VPU or FPGA, the Inference Engine uses the asynchronous approach (see
[Integrating Inference Engine in Your Application (current API)](Integrate_with_customer_application_new_API.md)).
The point is amortizing the costs of data transfers, by pipe-lining, see [Async API explained](@ref omz_demos_object_detection_demo_ssd_async_README).
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
Note that if you have more than one input, running as many inference requests as you have NUMA nodes (or sockets)
usually gives the same best latency as a single request on the single socket, but much higher throughput. Assuming two NUMA nodes machine:
```
$ benchmark_app -m <model.xml> -nstreams 2
 ```
Number of NUMA nodes on the machine can be queried via 'lscpu'.
Please see more on the NUMA support in the [Optimization Guide](supported_plugins/MULTI.md).

## Throughput Mode for CPU
Unlike most accelerators, CPU is perceived as an inherently latency-oriented device. 
Since 2018 R5 release, the Inference Engine introduced the "throughput" mode, which allows the Inference Engine to efficiently run multiple inference requests on the CPU simultaneously, greatly improving the throughput.

Internally, the execution resources are split/pinned into execution "streams".
Using this feature gains much better performance for the networks that originally are not scaled well with a number of threads (for example, lightweight topologies). This is especially pronounced for the many-core server machines.

Run the [Benchmark App](../../inference-engine/samples/benchmark_app/README.md) and play with number of infer requests running in parallel, next section. 
Try different values of the `-nstreams` argument from `1` to a number of CPU cores and find one that provides the best performance. 

In addition to the number of streams, it is also possible to play with the batch size to find the throughput sweet-spot.

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

## Kernels Tuning for GPU

GPU backend comes with a feature, that allows models tuning, so the workload is configured to fit better into hardware.

Tuning is time consuming process, which internally execute every layer several (or even hundreds) times to find most performant configuration.

This configuration is saved into json-formatted file, whose name can be passed as plugin param to network. GPU backend will process this data to configure kernels for the best performance.

For more details about Kernels Tuning and How-To please refer to [GPU Kernels Tuning](GPU_Kernels_Tuning.md). 
