# Deployment Optimization Guide {#openvino_docs_deployment_optimization_guide_dldt_optimization_guide}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:
   
   openvino_docs_deployment_optimization_guide_dldt_optimization_guide_additional

@endsphinxdirective

To optimize your inference performance consider the following: 

* Inputs Pre-processing with the OpenVINO

* Performance hints: Latency and Throughput

* Async API 

* Device-specific optimization 

* Combination of devices 

## Inputs Pre-processing with OpenVINO

In many cases, a network expects a pre-processed image, so make sure you do not perform unnecessary steps in your code:
- Model Optimizer can efficiently bake the mean and normalization (scale) values into the model (for example, to the weights of the first convolution). Please see [Model Optimizer](../MO_DG/prepare_model/Additional_Optimizations.md).
- Let the OpenVINO accelerate other means of [Image Pre-processing and Conversion](../OV_Runtime_UG/preprocessing_overview.md).
Note that in many cases, you can directly share the (input) data with the Inference Engine.

## Performance Hints: Throughput and Latency

One way to increase computational efficiency is batching, which combines many (potentially tens) of input images to achieve optimal throughput. 



### Throughput Mode for CPU: Internals <a name="cpu-streams"></a>
SELECTING THE #streams
Unlike most accelerators, CPU is perceived as an inherently latency-oriented device.
In fact, the OpenVINO does support the "throughput" mode for the CPU, which allows the Inference Engine to efficiently run multiple inference requests on the CPU simultaneously, greatly improving the overall throughput.

Internally, the execution resources are split/pinned into execution "streams".
This feature usually provides much better performance for the networks than batching. This is especially true for the many-core server machines:
![](../img/cpu_streams_explained_1.png)

Compared with the batching, the parallelism is somewhat transposed (i.e. performed over inputs, and much less within CNN ops):
![](../img/cpu_streams_explained.png)

Try the [Benchmark App](../../samples/cpp/benchmark_app/README.md) sample and play with number of streams running in parallel. The rule of thumb is tying up to a number of CPU cores on your machine.
For example, on an 8-core CPU, compare the `-nstreams 1` (which is a legacy, latency-oriented scenario) to the 2, 4, and 8 streams.

In addition, you can play with the batch size to find the throughput sweet spot.

If your application is hard or impossible to change in accordance with the multiple-requests logic, consider the "multiple-instance" trick to improve the throughput:  
-   For multi-socket execution, it is recommended to set   [`KEY_CPU_THREADS_NUM`](../OV_Runtime_UG/supported_plugins/CPU.md) to the number of cores per socket, and run as many instances of the application as you have sockets.
-   Similarly, for extremely lightweight networks (running faster than 1ms) and/or many-core machines (16+ cores), try limiting the number of CPU inference  threads to just `#&zwj;phys` cores and further, while trying to saturate the machine with running multiple instances of the application.

With CPU streams, the execution resources (CPU cores) are split/pinned into execution *streams*. Using this feature gains much better performance for the networks that originally are not scaled well with a number of threads (for example, lightweight topologies). This is especially pronounced for the many-core server machines. 

????
Run the Benchmark App and play with number of infer requests running in parallel, next section. Try different values of the -nstreams argument from 1 to a number of CPU cores and find one that provides the best performance. 

The throughput mode relaxes the requirement to saturate the CPU by using a large batch: running multiple independent inference requests in parallel often gives much better performance, than using a batch only. This allows you to simplify the app-logic, as you don't need to combine multiple inputs into a batch to achieve good CPU performance. Instead, it is possible to keep a separate infer request per camera or another source of input and process the requests in parallel using Async API. 

### Automatic Batching Internals
TBD

## Inference Engine Async API 

Inference Engine Async API can improve overall frame rate of the application. While accelerator is busy with the inference, the application can continue doing things on the host rather than wait for the inference to complete.

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

## Request-Based API and “GetBlob” Idiom <a name="new-request-based-api"></a>

Infer Request based API offers two types of request: Sync and Async. The Sync is considered below. The Async splits (synchronous) `Infer` into `StartAsync` and `Wait` (see <a href="#ie-async-api">Inference Engine Async API</a>).

More importantly, an infer request encapsulates the reference to the “executable” network and actual inputs/outputs. Now, when you load the network to the plugin, you get a reference to the executable network (you may consider that as a queue). Actual infer requests are created by the executable network:

```sh

@snippet snippets/dldt_optimization_guide6.cpp part6
```

`GetBlob` is a recommend way to communicate with the network, as it internally allocates the data with right padding/alignment for the device. For example, the GPU inputs/outputs blobs are mapped to the host (which is fast) if the `GetBlob` is used. But if you called the `SetBlob`, the copy (from/to the blob you have set) into the internal GPU plugin structures will happen.

## Additional GPU Checklist <a name="gpu-checklist"></a>

Inference Engine relies on the [Compute Library for Deep Neural Networks (clDNN)](https://01.org/cldnn) for Convolutional Neural Networks acceleration on Intel&reg; GPUs. Internally, clDNN uses OpenCL&trade; to implement the kernels. Thus, many general tips apply:

-	Prefer `FP16` over `FP32`, as the Model Optimizer can generate both variants and the `FP32` is default.
- 	Try to group individual infer jobs by using [automatic batching](../OV_Runtime_UG/automatic_batching.md)
-	Notice that using the GPU introduces one-time overhead (order of few seconds) of compiling the OpenCL kernels. The compilation happens upon loading the network to the GPU plugin and does not affect the inference time.
-	If your application is simultaneously using the inference on the CPU or otherwise loads the host heavily, make sure that the OpenCL driver threads do not starve. You can use [CPU configuration options](../OV_Runtime_UG/supported_plugins/CPU.md) to limit number of inference threads for the CPU plugin.
-	In the GPU-only scenario, a GPU driver might occupy a CPU core with spin-looped polling for completion. If the _CPU_ utilization is a concern, consider the `KEY_CLDND_PLUGIN_THROTTLE` configuration option.

> **NOTE**: See the [Benchmark App Sample](../../samples/cpp/benchmark_app/README.md) code for a usage example. 
Notice that while disabling the polling, this option might reduce the GPU performance, so usually this option is used with multiple [GPU streams](../OV_Runtime_UG/supported_plugins/GPU.md). 

## Heterogeneity <a name="heterogeneity"></a>

Heterogeneous execution (constituted by the dedicated Inference Engine [“Hetero” device](../OV_Runtime_UG/hetero_execution.md)) enables to schedule a network inference to the multiple devices.

### Typical Heterogeneous Scenarios of Concern <a name="heterogeneous-scenarios-of-concern"></a>

The primary points for executing a network in heterogeneous mode are as follows:

-	Calculate the heaviest pieces of the network with an accelerator while falling back to the CPU for the layers that are not supported by the accelerator.<br>
	This is particularly useful when certain custom (user) kernels are implemented only for the CPU (and much harder or even impossible to implement for the accelerator).

-	Use all available compute devices more efficiently, for example, by running branches of the network on the different devices.

### Heterogeneous Flow <a name="heterogeneous-flow"></a>

The execution through heterogeneous plugin has three distinct steps:

1.	**Applying affinity setting for the layers**, that is, binding them to the devices.

	-	This can be done automatically using *fallback priorities*, or on the *per-layer* basis.

	-	The affinity setting is made before loading the network to the (heterogeneous) plugin, so this is always a **static** setup with respect to execution.

2.	**Loading a network to the heterogeneous plugin**, which internally splits the network into subgraphs.<br>
	You can check the decisions the plugin makes, see <a href="#analyzing-heterogeneous-execution">Analysing the Heterogeneous Execution</a>.

3.	**Executing the infer requests**. From user’s side, this looks identical to a single-device case, while internally, the subgraphs are executed by actual plugins/devices.

Performance benefits of the heterogeneous execution depend heavily on the communications granularity between devices. If transmitting/converting data from one part device to another takes more time than the execution, the heterogeneous approach makes little or no sense. Using Intel&reg; VTune&trade; helps to visualize the execution flow on a timeline (see <a href="#vtune-examples">Intel&reg; VTune&trade; Examples</a>).

Similarly, if there are too much subgraphs, the synchronization and data transfers might eat the entire performance. In some cases, you can define the (coarser) affinity manually to avoid sending data back and forth many times during one inference.

The general affinity “rule of thumb” is to keep computationally-intensive kernels on the accelerator, and "glue" or helper  kernels on the CPU. Notice that this includes the granularity considerations. For example, running some custom activation (that comes after every accelerator-equipped convolution) on the CPU might result in performance degradation due to too much data type and/or layout conversions, even though the activation itself can be extremely fast. In this case, it might make sense to consider implementing the kernel for the accelerator (see <a href="#optimizing-custom-kernels">Optimizing Custom Kernels</a>). The conversions typically manifest themselves as outstanding (comparing to CPU-only execution) 'Reorder' entries (see <a href="#performance-counters">Internal Inference Performance Counters</a>).

For general details on the heterogeneous mode, refer to the [Heterogeneous execution guide](../OV_Runtime_UG/hetero_execution.md).

### Trying the Heterogeneous Plugin with Inference Engine Samples <a name="heterogeneous-plugin-with-samples"></a>

Every Inference Engine sample supports the `-d` (device) option.

For example, here is a command to run an [Classification Sample Async](../../samples/cpp/classification_sample_async/README.md):

```sh
./classification_sample_async -m <path_to_model>/Model.xml -i <path_to_pictures>/picture.jpg -d HETERO:GPU,CPU
```

where:

-	`HETERO` stands for Heterogeneous plugin.
-	`GPU,CPU` points to fallback policy with first priority on GPU and further fallback to CPU.

You can point more than two devices: `-d HETERO:HDDL,GPU,CPU`.

### General Tips on GPU/CPU Execution <a name="tips-on-gpu-cpu-execution"></a>

The following tips are provided to give general guidance on optimizing execution on GPU/CPU devices.

-	Generally, GPU performance is better on heavy kernels (like Convolutions) and large inputs. So if the network inference time is already too small (~1ms of execution time), using the GPU would unlikely give a boost.

-	A typical strategy to start with is to test the CPU-only and GPU-only scenarios first (with samples this is plain `-d CPU` or `-d GPU`). If there are specific kernels that are not supported by the GPU, the best option to try is the `HETERO:GPU,CPU` that automatically applies default splitting (based on the plugins layers support). Then, you can play with the manual affinity settings (for example, to further minimize the number of subgraphs).  

-	The general affinity “rule of thumb” is to keep computationally-intensive kernels on the accelerator, and "glue" (or helper) kernels on the CPU. Notice that this includes the granularity considerations. For example, running some (custom) activation on the CPU would result in too many conversions.

-	It is advised to do <a href="#analyzing-hetero-execution">performance analysis</a> to determine “hotspot” kernels, which should be the first candidates for offloading. At the same time, it is often more efficient to offload some reasonably sized sequence of kernels, rather than individual kernels, to minimize scheduling and other run-time overheads.

-	Notice that GPU can be busy with other tasks (like rendering). Similarly, the CPU can be in charge for the general OS routines and other application threads (see <a href="#note-on-app-level-threading">Note on the App-Level Threading</a>). Also, a high interrupt rate due to many subgraphs can raise the frequency of the one device and drag the frequency of another down.

-	Device performance can be affected by dynamic frequency scaling. For example, running long kernels on both devices simultaneously might eventually result in one or both devices stopping use of the Intel&reg; Turbo Boost Technology. This might result in overall performance decrease, even comparing to single-device scenario.

-	Mixing the `FP16` (GPU) and `FP32` (CPU) execution results in conversions and, thus, performance issues. If you are seeing a lot of heavy outstanding (compared to the CPU-only execution) Reorders, consider implementing actual GPU kernels. Refer to <a href="#performance-counters">Internal Inference Performance Counters</a> for more information.

### Analyzing Heterogeneous Execution <a name="analyzing-heterogeneous-execution"></a>

There is a dedicated configuration option that enables dumping the visualization of the subgraphs created by the heterogeneous mode, please see code example in the [Heterogeneous execution guide](../OV_Runtime_UG/hetero_execution.md)

After enabling the configuration key, the heterogeneous plugin generates two files:

-	`hetero_affinity.dot` - per-layer affinities. This file is generated only if default fallback policy was executed (as otherwise you have set the affinities by yourself, so you know them).
-	`hetero_subgraphs.dot` - affinities per sub-graph. This file is written to the disk during execution of `Core::LoadNetwork` for the heterogeneous flow.

You can use GraphViz\* utility or `.dot` converters (for example, to `.png` or `.pdf`), like xdot\*, available on Linux\* OS with `sudo apt-get install xdot`. 

You can also use performance data (in the [Benchmark App](../../samples/cpp/benchmark_app/README.md), it is an option `-pc`) to get performance data on each subgraph. Again, refer to the [Heterogeneous execution guide](../OV_Runtime_UG/hetero_execution.md) and to <a href="#performance-counters">Internal Inference Performance Counters</a> for a general counters information.

## Multi-Device Execution <a name="multi-device-optimizations"></a>
OpenVINO&trade; toolkit supports automatic multi-device execution, please see [Multi-Device execution](../OV_Runtime_UG/multi_device.md) description.
In the next chapter you can find the device-specific tips, while this section covers few recommendations 
for the multi-device execution:
-	MULTI usually performs best when the fastest device is specified first in the list of the devices. 
    This is particularly important when the parallelism is not sufficient 
    (e.g. the number of request in the flight is not enough to saturate all devices).
- It is highly recommended to query the optimal number of inference requests directly from the instance of the ExecutionNetwork 
  (resulted from the LoadNetwork call with the specific multi-device configuration as a parameter). 
Please refer to the code of the [Benchmark App](../../samples/cpp/benchmark_app/README.md) sample for details.    
-   Notice that for example CPU+GPU execution performs better with certain knobs 
    which you can find in the code of the same [Benchmark App](../../samples/cpp/benchmark_app/README.md) sample.
    One specific example is disabling GPU driver polling, which in turn requires multiple GPU streams (which is already a default for the GPU) to amortize slower 
    inference completion from the device to the host.
-	Multi-device logic always attempts to save on the (e.g. inputs) data copies between device-agnostic, user-facing inference requests 
    and device-specific 'worker' requests that are being actually scheduled behind the scene. 
    To facilitate the copy savings, it is recommended to start the requests in the order that they were created 
    (with ExecutableNetwork's CreateInferRequest).
  
Refer to [Deployment Optimization Guide Additional Configurations](dldt_deployment_optimization_guide_additional.md) to read more about performance during deployment step and learn about threading, working with multi-socket CPUs and Basic Interoperability with Other APIs.
