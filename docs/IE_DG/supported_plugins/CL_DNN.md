GPU Plugin {#openvino_docs_IE_DG_supported_plugins_CL_DNN}
=======

The GPU plugin uses the Intel&reg; Compute Library for Deep Neural Networks ([clDNN](https://01.org/cldnn)) to infer deep neural networks.
clDNN is an open source performance library for Deep Learning (DL) applications intended for acceleration of Deep Learning Inference on Intel&reg; Processor Graphics including Intel&reg; HD Graphics and Intel&reg; Iris&reg; Graphics.
For an in-depth description of clDNN, see: [clDNN sources](https://github.com/intel/clDNN) and [Accelerate Deep Learning Inference with Intel&reg; Processor Graphics](https://software.intel.com/en-us/articles/accelerating-deep-learning-inference-with-intel-processor-graphics).

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
* SimplerNMS
* PriorBox
* DetectionOutput

## Known Layers Limitations
* ROIPooling is supported for 'max' value of 'method' attribute.

## Supported Configuration Parameters

The plugin supports the configuration parameters listed below.
All parameters must be set before calling <code>InferenceEngine::Core::LoadNetwork()</code> in order to take effect.
When specifying key values as raw strings (that is, when using Python API), omit the `KEY_` prefix.

| Parameter Name          | Parameter Values                | Default         | Description                                               |
|---------------------|-----------------------------|-----------------|-----------------------------------------------------------|
| `KEY_PERF_COUNT`      | `YES` / `NO`                    | `NO`              | Collect performance counters during inference             |
| `KEY_CONFIG_FILE`     | `"<file1> [<file2> ...]"`         | `""`              | Load custom layer configuration files                     |
| `KEY_DUMP_KERNELS`    | `YES` / `NO`                    | `NO`              | Dump the final kernels used for custom layers             |
| `KEY_TUNING_MODE`     | `TUNING_DISABLED` <br /> `TUNING_CREATE` <br />  `TUNING_USE_EXISTING`            | `TUNING_DISABLED` | Disable inference kernel tuning     <br /> Create tuning file (expect much longer runtime)  <br />         Use an existing tuning file              |
| `KEY_TUNING_FILE`     | `"<filename>"`                  | `""`              | Tuning file to create / use                               |
| `KEY_CLDNN_PLUGIN_PRIORITY` | `<0-3>`                       | `0`               | OpenCL queue priority (before usage, make sure your OpenCL driver supports appropriate extension)<br> Higher value means higher priority for clDNN OpenCL queue. 0 disables the setting. |
| `KEY_CLDNN_PLUGIN_THROTTLE` | `<0-3>`                       | `0`               | OpenCL queue throttling (before usage, make sure your OpenCL driver supports appropriate extension)<br> Lower value means lower driver thread priority and longer sleep time for it. 0 disables the setting. |
| `KEY_CLDNN_GRAPH_DUMPS_DIR` | `"<dump_dir>"`                       | `""`               | clDNN graph optimizer stages dump output directory (in GraphViz format)                                     |
| `KEY_CLDNN_SOURCES_DUMPS_DIR` | `"<dump_dir>"`                       | `""`               | Final optimized clDNN OpenCL sources dump output directory                                   |
| `KEY_GPU_THROUGHPUT_STREAMS`  | `KEY_GPU_THROUGHPUT_AUTO`, or positive integer| 1 | Specifies a number of GPU "execution" streams for the throughput mode (upper bound for a number of inference requests that can be executed simultaneously).<br>This option is can be used to decrease GPU stall time by providing more effective load from several streams. Increasing the number of streams usually is more effective for smaller topologies or smaller input sizes. Note that your application should provide enough parallel slack (e.g. running many inference requests) to leverage full GPU bandwidth. Additional streams consume several times more GPU memory, so make sure the system has enough memory available to suit parallel stream execution. Multiple streams might also put additional load on CPU. If CPU load increases, it can be regulated by setting an appropriate `KEY_CLDNN_PLUGIN_THROTTLE` option value (see above). If your target system has relatively weak CPU, keep throttling low. <br>The default value is 1, which implies latency-oriented behaviour.<br>`KEY_GPU_THROUGHPUT_AUTO` creates bare minimum of streams to improve the performance; this is the most portable option if you are not sure how many resources your target machine has (and what would be the optimal number of streams). <br> A positive integer value creates the requested number of streams. |
| `KEY_EXCLUSIVE_ASYNC_REQUESTS` | `YES` / `NO`                | `NO`              | Forces async requests (also from different executable networks) to execute serially.|

## Note on Debug Capabilities of the GPU Plugin

Inference Engine GPU plugin provides possibility to dump the user custom OpenCL&trade; kernels to a file to allow you to properly debug compilation issues in your custom kernels.

The application can use the <code>SetConfig()</code> function with the key <code>PluginConfigParams::KEY_DUMP_KERNELS</code> and value: <code>PluginConfigParams::YES</code>. Then during network loading, all custom layers will print their OpenCL kernels with the JIT instrumentation added by the plugin.
The kernels will be stored in the working directory under files named the following way: <code>clDNN_program0.cl</code>, <code>clDNN_program1.cl</code>.

This option is disabled by default. Additionally, the application can call the <code>SetConfig()</code> function with the key <code>PluginConfigParams::KEY_DUMP_KERNELS</code> and value: <code>PluginConfigParams::NO</code> before network loading.

How to verify that this option is disabled:
1.  Delete all <code>clDNN_program*.cl</code> files from the current directory
2.  Run your application to load a network
3.  Examine the working directory for the presence of any kernel file (for example, <code>clDNN_program0.cl</code>)

## GPU Context and Video Memory Sharing RemoteBlob API

See [RemoteBlob API of GPU Plugin](GPU_RemoteBlob_API.md)

## See Also
* [Supported Devices](Supported_Devices.md)

[conv_simple_01]: ../img/conv_simple_01.png
[pooling_fakequant_01]: ../img/pooling_fakequant_01.png
[fullyconnected_activation_01]: ../img/fullyconnected_activation_01.png
[group_convolutions_01]: ../img/group_convolutions_01.png
[conv_sum_relu_01]: ../img/conv_sum_relu_01.png
