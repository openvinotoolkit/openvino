# GPU Plugin {#openvino_docs_IE_DG_supported_plugins_GPU}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_IE_DG_supported_plugins_GPU_RemoteBlob_API


@endsphinxdirective

The GPU plugin uses the Intel® Compute Library for Deep Neural Networks (clDNN) to infer deep neural networks.
clDNN is an open source performance library for Deep Learning (DL) applications intended for acceleration of Deep Learning Inference on Intel® Processor Graphics including Intel® HD Graphics, Intel® Iris® Graphics, Intel® Iris® Xe Graphics, and Intel® Iris® Xe MAX graphics.
For an in-depth description of clDNN, see [Inference Engine source files](https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_gpu/) and [Accelerate Deep Learning Inference with Intel® Processor Graphics](https://software.intel.com/en-us/articles/accelerating-deep-learning-inference-with-intel-processor-graphics).

## Device Naming Convention
* Devices are enumerated as "GPU.X" where `X={0, 1, 2,...}`. Only Intel® GPU devices are considered.
* If the system has an integrated GPU, it always has id=0 ("GPU.0").
* Other GPUs have undefined order that depends on the GPU driver.
* "GPU" is an alias for "GPU.0"
* If the system doesn't have an integrated GPU, then devices are enumerated starting from 0.

For demonstration purposes, see the [Hello Query Device C++ Sample](../../../samples/cpp/hello_query_device/README.md) that can print out the list of available devices with associated indices. Below is an example output (truncated to the device names only):

```sh
./hello_query_device
Available devices:
    Device: CPU
...
    Device: GPU.0
...
    Device: GPU.1
...
    Device: HDDL
```

## Optimizations

The plugin supports algorithms that fuse several operations into one optimized operation. Refer to the sections below for details.

> **NOTE**: For operation descriptions, see the [IR Notation Reference](../../ops/opset.md).

### Fusing Convolution and Simple Layers

Merge of a Convolution layer and any of the simple layers listed below:
- Activation: ReLU, ELU, Sigmoid, Clamp, and others
- Depthwise: ScaleShift, PReLU
- FakeQuantize

> **NOTE**: You can have any number and order of simple layers.

A combination of a Convolution layer and simple layers results in a single fused layer called
*Convolution*:
![conv_simple_01]


### Fusing Pooling and FakeQuantize Layers

A combination of Pooling and FakeQuantize layers results in a single fused layer called *Pooling*:
![pooling_fakequant_01]

### Fusing Activation Layers

Given the linear pattern, an Activation layer can be fused into other layers:

![fullyconnected_activation_01]


### Fusing Convolution and Sum Layers

A combination of Convolution, Simple, and Eltwise layers with the sum operation results in a single layer called  *Convolution*:
![conv_sum_relu_01]

### Fusing a Group of Convolutions

If a topology contains the following pipeline, a GPU plugin merges Split, Convolution, and Concatenation layers  into a single Convolution layer with the group parameter:
> **NOTE**: Parameters of the Convolution layers must coincide.

![group_convolutions_01]

### Optimizing Layers Out

The following layers are optimized out under certain conditions:
  * Crop
  * Concatenate
  * Reshape
  * Flatten
  * Split
  * Copy

### Load-Time Execution

Some layers are executed during the load time, not during the inference. One of such layers is PriorBox.


## CPU Executed Layers

The following layers are not accelerated on the GPU and executed on the host CPU instead:
* Proposal
* NonMaxSuppression
* PriorBox
* DetectionOutput

## Supported Configuration Parameters

The plugin supports the configuration parameters listed below.
All parameters must be set before calling <code>InferenceEngine::Core::LoadNetwork()</code> in order to take effect.
When specifying key values as raw strings (that is, when using Python API), omit the `KEY_` prefix.

| Parameter Name          | Parameter Values                | Default         | Description                                               |
|---------------------|-----------------------------|-----------------|-----------------------------------------------------------|
| `KEY_CACHE_DIR`      | `"<cache_dir>"`                    | `""`              | Specifies a directory where compiled OCL binaries can be cached. First model loading generates the cache, and all subsequent LoadNetwork calls use precompiled kernels which significantly improves load time. If empty - caching is disabled             |
| `KEY_PERF_COUNT`      | `YES` / `NO`                    | `NO`              | Collect performance counters during inference             |
| `KEY_CONFIG_FILE`     | `"<file1> [<file2> ...]"`         | `""`              | Load custom layer configuration files                     |
| `KEY_GPU_HOST_`<br>`TASK_PRIORITY` | `GPU_HOST_TASK_PRIORITY_<HIGH\|MEDIUM\|LOW>`                       | `GPU_HOST_TASK_PRIORITY_MEDIUM`               | This key instructs the GPU plugin which cpu core type of TBB affinity used in load network. <br> This option has 3 types of levels: HIGH, LOW, and ANY. It is only affected on Hybrid CPUs. <br>- LOW - instructs the GPU Plugin to use LITTLE cores if they are available <br>- MEDIUM (DEFAULT) - instructs the GPU Plugin to use any available cores (BIG or LITTLE cores) <br>- HIGH - instructs the GPU Plugin to use BIG cores if they are available |
| `KEY_GPU_PLUGIN_`<br>`PRIORITY` | `<0-3>`                       | `0`               | OpenCL queue priority (before usage, make sure your OpenCL driver supports appropriate extension)<br> Higher value means higher priority for OpenCL queue. 0 disables the setting. **Deprecated**. Please use KEY_GPU_MODEL_PRIORITY |
| `KEY_GPU_PLUGIN_`<br>`THROTTLE` | `<0-3>`                       | `2`               | OpenCL queue throttling (before usage, make sure your OpenCL driver supports appropriate extension)<br> Lower value means lower driver thread priority and longer sleep time for it. Has no effect if the driver does not support reqired hint.  |
| `KEY_CLDNN_ENABLE_`<br>`FP16_FOR_QUANTIZED_`<br>`MODELS` | `YES` / `NO`                       | `YES`               | Allows using FP16+INT8 mixed precision mode, so non-quantized parts of a model will be executed in FP16 precision for FP16 IR. Does not affect quantized FP32 IRs |
| `KEY_GPU_NV12_`<br>`TWO_INPUTS` | `YES` / `NO`                       | `NO`               | Controls preprocessing logic for nv12 input. If it's set to YES, then device graph will expect that user will set biplanar nv12 blob as input wich will be directly passed to device execution graph. Otherwise, preprocessing via GAPI is used to convert NV12->BGR, thus GPU graph have to expect single input |
| `KEY_GPU_THROUGHPUT_`<br>`STREAMS`  | `KEY_GPU_THROUGHPUT_AUTO`, or positive integer| 1 | Specifies a number of GPU "execution" streams for the throughput mode (upper bound for a number of inference requests that can be executed simultaneously).<br>This option is can be used to decrease GPU stall time by providing more effective load from several streams. Increasing the number of streams usually is more effective for smaller topologies or smaller input sizes. Note that your application should provide enough parallel slack (e.g. running many inference requests) to leverage full GPU bandwidth. Additional streams consume several times more GPU memory, so make sure the system has enough memory available to suit parallel stream execution. Multiple streams might also put additional load on CPU. If CPU load increases, it can be regulated by setting an appropriate `KEY_GPU_PLUGIN_THROTTLE` option value (see above). If your target system has relatively weak CPU, keep throttling low. <br>The default value is 1, which implies latency-oriented behavior.<br>`KEY_GPU_THROUGHPUT_AUTO` creates bare minimum of streams to improve the performance; this is the most portable option if you are not sure how many resources your target machine has (and what would be the optimal number of streams). <br> A positive integer value creates the requested number of streams. |
| `KEY_EXCLUSIVE_ASYNC_`<br>`REQUESTS` | `YES` / `NO`                | `NO`              | Forces async requests (also from different executable networks) to execute serially.|
| `KEY_GPU_MAX_NUM_`<br>`THREADS` | `integer value` | `maximum # of HW threads available in host environment` |  Specifies the number of CPU threads that can be used for GPU engine, e.g, JIT compilation of GPU kernels or cpu kernel processing within GPU plugin. The default value is set as the number of maximum available threads in host environment to minimize the time for LoadNetwork, where the GPU kernel build time occupies a large portion. Note that if the specified value is larger than the maximum available # of threads or less than zero, it is set as maximum available # of threads. It can be specified with a smaller number than the available HW threads according to the usage scenario, e.g., when the user wants to assign more CPU threads while GPU plugin is running. Note that setting this value with lower number will affect not only the network loading time but also the cpu layers of GPU networks that are optimized with multi-threading. |
| `KEY_GPU_ENABLE_`<br>`LOOP_UNROLLING` | `YES` / `NO`             | `YES`             | Enables recurrent layers such as TensorIterator or Loop with fixed iteration count to be unrolled. It is turned on by default. Turning this key on will achieve better inference performance for loops with not too many iteration counts (less than 16, as a rule of thumb). Turning this key off will achieve better performance for both graph loading time and inference time with many iteration counts (greater than 16). Note that turning this key on will increase the graph loading time in proportion to the iteration counts. Thus, this key should be turned off if graph loading time is considered to be most important target to optimize. |
| `KEY_CLDNN_PLUGIN_`<br>`PRIORITY` | `<0-3>`                       | `0`               | OpenCL queue priority (before usage, make sure your OpenCL driver supports appropriate extension)<br> Higher value means higher priority for OpenCL queue. 0 disables the setting. **Deprecated**. Please use KEY_GPU_MODEL_PRIORITY |
| `KEY_CLDNN_PLUGIN_`<br>`THROTTLE` | `<0-3>`                       | `0`               | OpenCL queue throttling (before usage, make sure your OpenCL driver supports appropriate extension)<br> Lower value means lower driver thread priority and longer sleep time for it. 0 disables the setting. **Deprecated**. Please use KEY_GPU_PLUGIN_THROTTLE |
| `KEY_CLDNN_GRAPH_`<br>`DUMPS_DIR` | `"<dump_dir>"`                       | `""`               | clDNN graph optimizer stages dump output directory (in GraphViz format) **Deprecated**. Will be removed in the next release                                     |
| `KEY_CLDNN_SOURCES_`<br>`DUMPS_DIR` | `"<dump_dir>"`                       | `""`               | Final optimized clDNN OpenCL sources dump output directory. **Deprecated**. Will be removed in the next release                                   |
| `KEY_DUMP_KERNELS`    | `YES` / `NO`                    | `NO`              | Dump the final kernels used for custom layers. **Deprecated**. Will be removed in the next release             |
| `KEY_TUNING_MODE`     | `TUNING_DISABLED` <br /> `TUNING_CREATE` <br />  `TUNING_USE_EXISTING`            | `TUNING_DISABLED` | Disable inference kernel tuning     <br /> Create tuning file (expect much longer runtime)  <br />         Use an existing tuning file. **Deprecated**. Will be removed in the next release |
| `KEY_TUNING_FILE`     | `"<filename>"`                  | `""`              | Tuning file to create / use. **Deprecated**. Will be removed in the next release |

## Quering GPU specific metric keys
* MEMORY_STATISTICS : Returns overall memory statistics of `GPU` device allocated by engine with allocation types. If the network has `TensorIterator` or `Loop` operation which is not unrolled, there will be additional allocation at the first inference phase. In such a case, querying for `MEMORY_STATISTICS` should be done after first inference for more accurate result. The code below demonstrates how to query overall memory statistics of `GPU` device:

@snippet snippets/GPU_Metric0.cpp part0

* MAX_BATCH_SIZE : Returns maximum batch size for a given network which is not only executable but also does not lose performance due to the memory swap impact. Note that the returned value may not aligned to power of 2. Also, MODEL_PTR is the required option for this metric since the available max batch size depends on the model size. If the MODEL_PTR is not given, it will return 1. The example code to set the required and optional configs for this metic is available in the following snippet:

@snippet snippets/GPU_Metric1.cpp part1

* OPTIMAL_BATCH_SIZE : Returns _optimal_ batch size for a given network on the given GPU device. The returned value is aligned to power of 2. Also, MODEL_PTR is the required option for this metric since the optimal batch size highly depends on the model. If the MODEL_PTR is not given, the value of 1 is returned. The example code to set the required and optional configs for this metric is available in the following snippet:

@snippet snippets/GPU_Metric1.cpp part2
## GPU Context and Video Memory Sharing RemoteBlob API

See [RemoteBlob API of GPU Plugin](GPU_RemoteBlob_API.md)

## See Also
* [Supported Devices](Supported_Devices.md)

[conv_simple_01]: ../img/conv_simple_01.png
[pooling_fakequant_01]: ../img/pooling_fakequant_01.png
[fullyconnected_activation_01]: ../img/fullyconnected_activation_01.png
[group_convolutions_01]: ../img/group_convolutions_01.png
[conv_sum_relu_01]: ../img/conv_sum_relu_01.png
