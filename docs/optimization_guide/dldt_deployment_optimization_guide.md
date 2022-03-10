# Deployment Optimization Guide {#openvino_docs_deployment_optimization_guide_dldt_optimization_guide}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:
   
@endsphinxdirective

As explained in the general [parent topic](./dldt_optimization_guide.md), there are model and runtime level optimizations. 
This document explains how to optimize your _runtime_ inference performance with the following options: 

* Inputs Pre-processing with the OpenVINO

* High-level Performance Presets (Hints): Throughput and Latency

* Async API and 'get_tensor' Idiom 

* Device-specific optimization 

* Combination of devices 

## Inputs Pre-processing with OpenVINO

In many cases, a network expects a pre-processed image, so make sure you do not perform unnecessary steps in your code:
- Model Optimizer can efficiently bake the mean and normalization (scale) values into the model (for example, to the weights of the first convolution). Please see [relevant Model Optimizer command-line options](../MO_DG/prepare_model/Additional_Optimizations.md).
- Let the OpenVINO accelerate other means of [Image Pre-processing and Conversion](../OV_Runtime_UG/preprocessing_overview.md).
- Note that in many cases, you can directly share the (input) data with the OpenVINO, for example consider [remote tensors API of the GPU Plugin](../OV_Runtime_UG//supported_plugins/GPU_RemoteBlob_API.md).

## High-level Performance Presets (Hints): Throughput and Latency
Traditionally, each of the OpenVINO's [supported devices](../OV_Runtime_UG/supported_plugins/Supported_Devices.md) offers a bunch of low-level performance settings. 
Tweaking this detailed configuration requires deep architecture understanding.
Also, while the resulting performance may be optimal for the specific combination of the device and the model that is inferred, it is actually neither device/model nor future-proof:
- Even within a family of the devices (like various CPUs), things like different number of CPU cores would eventually result in different execution configuration to be optimal.
- Similarly the optimal batch size is very much specific to the particular instance of the GPU.
- Compute vs memory-bandwidth requirements for the model being inferenced, as well as inference precision, possible model's quantization and other factors add more unknowns to the resulting performance equation.
- Finally, the optimal execution parameters of one device do not transparently map to another device type, for example:
 - - Both the CPU and GPU devices support the notion of the 'streams' (i.e. inference instances that are executed in parallel), yet the optimal number of the streams is deduced very differently.
 
Beyond execution _parameters_ there are potentially many device-specific details like _scheduling_ that greatly affect the performance. 
Specifically, GPU-oriented tricks like batching, which combines many (potentially tens) of input images to achieve optimal throughput, do not always map well to the CPU, as e.g. detailed in the next sections.
The hints allow to really hide _execution_ specifics required to saturate the device. For example, no need to explicitly combine multiple inputs into a batch to achieve good GPU performance.
Instead, it is possible to keep a separate infer request per camera or another source of input and process the requests in parallel using <a href="#ov-async-api">OpenVINO Async API</a>.
The only requirement for the application is about running multiple inference requests in parallel.
OpenVINO's device-specific implementation of the hints will take care of the rest. This allows a developer to greatly simplify the app-logic.

In summary, when the performance _portability_ is of concern, consider the [High-Level Performance Hints](../OV_Runtime_UG/performance_hints.md). 
Below you can find the implementation details (particularly how the OpenVINO implements the 'throughput' approach) for the specific devices. 
Keep in mind that while multiple scheduling and/or batching approaches (combining individual inference requests) can work together, the hints make these decisions to be transparent to the application.

### OpenVINO Streams <a name="cpu-streams"></a>
As detailed in the section <a href="#ov-async-api">OpenVINO Async API</a>) running multiple inference requests asynchronously is important for general application efficiency.
Additionally, most devices support running multiple inference requests in parallel in order to improve the device utilization. The _level_ of the parallelism (i.e. how many requests are really executed in parallel on the device) is commonly referred as a number of 'streams'.  Notice that for efficient asynchronous execution, the streams are actually handling inference with special pool of the threads.
So each time you start inference requests (potentially from different application threads), they are actually muxed into a inference queue of the particular `ov:compiled_model`.  
If there is a vacant stream, it pops the request from the queue and actually expedites that to the on-device execution.

### Throughput on the CPU: Internals <a name="cpu-streams"></a>
In order to best serve multiple inference requests simultaneously, the inference threads are grouped/pinned to the particular CPU cores, constituting execution "streams".
This provides much better performance for the networks than batching especially for the many-core machines:
![](../img/cpu_streams_explained_1.png)

Compared with the batching, the parallelism is somewhat transposed (i.e. performed over inputs, with much less synchronization within CNN ops):
![](../img/cpu_streams_explained.png)

Notice that [high-level performance hints](../OV_Runtime_UG/performance_hints.md) allows the implementation to select the optimal number of the streams, _depending on the model compute demands_ and CPU capabilities (including [int8 inference](../OV_Runtime_UG/Int8Inference.md) hardware acceleration, number of cores, etc). 

### Automatic Batching Internals <a name="ov-auto-batching"></a>
While the GPU plugin fully supports general notion of the streams, the associated performance improvements are usually modest.
The primary reason is that, while the streams allow to hide the communication overheads and hide certain bubbles in device utilization, running multiple OpenCL kernels on the GPU simultaneously is less efficient, compared to calling a kernel on the multiple inputs at once.   

As explained in the section on the [automatic batching](../OV_Runtime_UG/automatic_batching.md), the feature performs on-the-fly grouping of the inference requests to improve device utilization.
The Automatic Batching relaxes the requirement for an application to saturate devices like GPU by _explicitly_ using a large batch. It essentially it performs transparent inputs gathering from 
individual inference requests followed by the actual batched execution, with no programming effort from the user:
![](../img/BATCH_device.PNG)

Essentially, the Automatic Batching shifts the asynchronousity from the individual requests to the groups of requests that constitute the batches. Thus, for the execution to be efficient it is very important that the requests arrive timely, without causing a batching timeout. 
Normally, the timeout should never be hit. It is rather a graceful way to handle the application exit (when the inputs are not arriving anymore, so the full batch is not possible to collect). So if your workload experiences the timeouts (resulting in the performance drop, as the timeout value adds itself to the latency of every request), consider balancing the timeout value vs the batch size. For example in many cases having smaller timeout value/batch size may yield better performance than large batch size, but coupled with the timeout value that is cannot guarantee accommodating the full number of the required requests.

## OpenVINO Async API <a name="ov-async-api"></a>

OpenVINO Async API can improve overall throughput rate of the application. While a device is busy with the inference, the application can do other things in parallel (e.g. populating inputs or scheduling other requests) rather than wait for the inference to complete.

In the example below, inference is applied to the results of the video decoding. So it is possible to keep two parallel infer requests, and while the current is processed, the input frame for the next is being captured. This essentially hides the latency of capturing, so that the overall frame rate is rather determined only by the slowest part of the pipeline (decoding IR inference) and not by the sum of the stages.

You can compare the pseudo-codes for the regular and async-based approaches:

-	In the regular way, the frame is captured with OpenCV and then immediately processed:<br>

@snippet snippets/dldt_optimization_guide8.cpp part8

![Intel&reg; VTune&trade; screenshot](../img/vtune_regular.png)

-	In the "true" async mode, the `NEXT` request is populated in the main (application) thread, while the `CURRENT` request is processed:<br>

@snippet snippets/dldt_optimization_guide9.cpp part9

![Intel&reg; VTune&trade; screenshot](../img/vtune_async.png)

The technique can be generalized to any available parallel slack. For example, you can do inference and simultaneously encode the resulting or previous frames or run further inference, like emotion detection on top of the face detection results.
Refer to the [Object Detection С++ Demo](@ref omz_demos_object_detection_demo_cpp), [Object Detection Python Demo](@ref omz_demos_object_detection_demo_python)(latency-oriented Async API showcase) and [Benchmark App Sample](../../samples/cpp/benchmark_app/README.md) for complete examples of the Async API in action.

The API of the inference requests offers Sync and Async execution. While the `ov::InferRequest::infer()` is inherently synchronous and executes immediately (effectively serializing the execution flow in the current application thread), the Async "splits" the `infer()` into `ov::InferRequest::start_async()` and `ov::InferRequest::wait()`. Please consider the [API examples](../OV_Runtime_UG/ov_infer_request.md).

A typical use-case for the `ov::InferRequest::infer()` is running a dedicated application thread per source of inputs (e.g. a camera), so that every step (frame capture, processing, results parsing and associated logic) is kept serial within the thread.
In contrast, the `ov::InferRequest::start_async()` and `ov::InferRequest::wait()` allow the application to continue its activities and poll or wait for the inference completion. So one reason for using asynchronous code is _efficiency_.

Notice that the `ov::InferRequest::wait()` waits for the specific request only. However, running multiple inference requests in parallel provides no guarantees on the completion order. This may complicate a possible logic based on the `ov::InferRequest::wait`. The most scalable approach is using callbacks (set via the `ov::InferRequest::set_callback`) that are executed upon completion of the request. The callback functions will be used by the OpenVINO runtime to notify on the results (or errors). 
This is more event-driven approach for you to implement the flow control.

Few important points on the callbacks:
- It is the application responsibility to ensure that any callback function is thread-safe
- Although executed asynchronously by a dedicated threads the callbacks should NOT include heavy operations (e.g. I/O) and/or blocking calls. Keep the work done by any callback to a minimum.

## "get_tensor" Idiom <a name="new-request-based-api"></a>

`get_tensor` is a recommended way to populate the inference inputs (and read back the outputs), as it internally allocates the data with right padding/alignment for the device. For example, the GPU inputs/outputs tensors are mapped to the host (which is fast) only when the `get_blob` is used, while for the `set_tensor` a copy into the internal GPU structures may happen.
Please consider the [API examples](../OV_Runtime_UG/ov_infer_request.md).
In contrast, the `set_tensor` is a preferable way to handle [remote tensors for example with the GPU device](../OV_Runtime_UG//supported_plugins/GPU_RemoteBlob_API.md).

## Additional GPU Checklist <a name="gpu-checklist"></a>

OpenVINO relies on the OpenCL&trade; kernels for the GPU implementation. Thus, many general OpenCL tips apply:

-	Prefer `FP16` inference precision over `FP32`, as the Model Optimizer can generate both variants and the `FP32` is default. Also, consider [int8 inference](../OV_Runtime_UG/Int8Inference.md)
- 	Try to group individual infer jobs by using [automatic batching](../OV_Runtime_UG/automatic_batching.md)
-	Consider [caching](../OV_Runtime_UG/Model_caching_overview.md) to minimize model load time
-	If your application is simultaneously using the inference on the CPU or otherwise loads the host heavily, make sure that the OpenCL driver threads do not starve. You can use [CPU configuration options](../OV_Runtime_UG/supported_plugins/CPU.md) to limit number of inference threads for the CPU plugin.
-	Even in the GPU-only scenario, a GPU driver might occupy a CPU core with spin-looped polling for completion. If the _CPU_ utilization is a concern, consider the dedicated [throttling configuration option](../OV_Runtime_UG/supported_plugins/GPU.md). Notice that this option might increase the inference latency, so consider combining with multiple [GPU streams](../OV_Runtime_UG/supported_plugins/GPU.md) or [throughput performance hints](../OV_Runtime_UG/performance_hints.md).
- When operating media inputs consider [remote tensors API of the GPU Plugin](../OV_Runtime_UG//supported_plugins/GPU_RemoteBlob_API.md).

## Multi-Device Execution <a name="multi-device-optimizations"></a>
OpenVINO&trade; toolkit supports automatic multi-device execution, please see [Multi-Device execution](../OV_Runtime_UG/multi_device.md) description. 
This section covers few recommendations for the multi-device execution:
- MULTI usually performs best when the fastest device is specified first in the list of the devices. 
    This is particularly important when the request-level parallelism is not sufficient 
    (e.g. the number of request in the flight is not enough to saturate all devices).
- Just like with any throughput-oriented execution, it is highly recommended to query the optimal number of inference requests directly from the instance of the `ov:compiled_model`. 
Please refer to the code of the [Benchmark App](../../samples/cpp/benchmark_app/README.md) sample for details.    
-   Notice that for example CPU+GPU execution performs better with certain knobs 
    which you can find in the code of the same [Benchmark App](../../samples/cpp/benchmark_app/README.md) sample.
    One specific example is disabling GPU driver polling, which in turn requires multiple GPU streams to amortize slower 
    communication of inference completion from the device to the host.
-	Multi-device logic always attempts to save on the (e.g. inputs) data copies between device-agnostic, user-facing inference requests 
    and device-specific 'worker' requests that are being actually scheduled behind the scene. 
    To facilitate the copy savings, it is recommended to run the requests in the order that they were created.

### Internal Inference Performance Counters and Execution Graphs <a name="performance-counters"></a>

Both [C++](../../samples/cpp/benchmark_app/README.md) and [Python](../../tools/benchmark_tool/README.md) versions of the `benchmark_app` supports a `-pc` command-line parameter that outputs internal execution breakdown.

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