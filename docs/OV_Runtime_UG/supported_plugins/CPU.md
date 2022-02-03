# CPU device {#openvino_docs_OV_UG_supported_plugins_CPU}


The plugin allows to achieve high performance of inference using CPU, using the Intel® Math Kernel Library for Deep Neural Networks (Intel® MKL-DNN). It utilizes Intel® Threading Building Blocks (Intel® TBB) to parallelize calculations. For performance considerations, please refer to the [Optimization Guide](../../optimization_guide/dldt_optimization_guide.md).

Its set of supported layers can be expanded with [the Extensibility mechanism](../Extensibility_DG/Intro.md).

The CPU plugin supports inference on:
- Intel® Xeon® with Intel® Advanced Vector Extensions 2 (Intel® AVX2),
- Intel® Advanced Vector Extensions 512 (Intel® AVX-512) and AVX512_BF16,
- Intel® Core™ Processors with Intel® AVX2, 
- Intel Atom® Processors with Intel® Streaming SIMD Extensions (Intel® SSE).

To see which configuration is used by a layer, you can use the `-pc` flag for samples. 
It shows execution statistics that you can use to get information about layer name, layer type, 
execution status, execution time, and the type of the execution primitive.

**First inference latency**  
To reduce first inference latency, the plugin utilizes the ngraph function serialization mechanism to enable model caching. Caching a compiled blob eliminates the impact of Common Transformations and Low Precision Transformations on first inference latency of the following runs.


## Internal CPU Plugin Optimizations

The CPU plugin supports several graph optimization algorithms, such as fusing or removing layers.
For layer descriptions, see the [IR Notation Reference](../../ops/opset.md).

**For details, click on the optimization algorithm you are interested in.**  
@sphinxdirective
.. dropdown:: Lowering Inference Precision

   The CPU plugin follows the default optimization approach, which means that inference is done with lower precision if it is possible on the given platform to reach better performance with an acceptable range of accuracy. For details, see the [Using Bfloat16 Inference](../Bfloat16Inference.md).

.. dropdown:: Fusing Convolution and Simple Layers

   Merging of a convolution layer with any number of the following simple layers, in any order:
   - Activation: ReLU, ELU, Sigmoid, Clamp
   - Depthwise: ScaleShift, PReLU
   - FakeQuantize

   This combination results in a single fused layer called *Convolution*:

   .. image:: _images/conv_simple_01.png
      :width: 300px
      :align: center

.. dropdown:: Fusing Pooling and FakeQuantize Layers

   A combination of Pooling and FakeQuantize layers results in a single fused layer called *Pooling*:  

   .. image:: _images/pooling_fakequant_01.png
      :width: 300px
      :align: center

.. dropdown:: Fusing FullyConnected and Activation Layers

   A combination of FullyConnected and Activation layers results in a single fused layer called *FullyConnected*:

   .. image:: _images/fullyconnected_activation_01.png
      :width: 300px
      :align: center

.. dropdown:: Fusing Convolution and Depthwise Convolution Layers Grouped with Simple Layers
 
   A combination of two groups: a group of a Convolution or a Binary Convolution layer with simple layers, and a group of a Depthwise Convolution layer with simple layers. It results in a single layer called *Convolution* or *Binary Convolution*:

   .. note::
   
      This pattern is possible only on CPUs with support of Streaming SIMD Extensions 4.2 (SSE 4.2) and Intel AVX2 Instruction Set Architecture (ISA).
   
      Also, Depthwise convolution layers should have the same values for the `group`, input channels, and output channels parameters.

   .. image:: _images/conv_depth_01.png
     :width: 300px
     :align: center

.. dropdown:: Fusing Convolution and Sum Layers 

   A combination of convolution, simple, and Eltwise layers with the sum operation results in a single layer called *Convolution*:  

    .. image:: _images/conv_sum_relu_01.png
      :width: 300px
      :align: center

.. dropdown:: Fusing a Group of Convolutions

   If a topology contains the following pipeline and convolution layers' parameters coincide, a CPU plugin merges split, convolution, and concatenation layers into a single convolution layer with the group parameter:   

   .. image:: _images/group_convolutions_01.png
      :width: 300px
      :align: center

.. dropdown:: Removing a Power Layer

   CPU plugin removes a Power layer from a topology if it has the following parameters:
      - ``power = 1``
      - ``scale = 1``
      - ``offset = 0``

@endsphinxdirective

## Supported Configuration Parameters

The following are configuration parameters supported by the CPU plugin.
All parameters must be set with the `InferenceEngine::Core::LoadNetwork()` method.
When specifying key values as raw strings (that is, when using Python API), omit the `KEY_` prefix.
Refer to the OpenVINO samples for usage examples: [Benchmark App](../../../samples/cpp/benchmark_app/README.md).

@sphinxdirective

| **KEY_EXCLUSIVE_ASYNC_REQUESTS**
| Parameter values: YES/NO
| Default value: NO
| Description:
|    A general option also supported by other plugins. It forces async requests (also from different executable networks) to execute serially. This prevents potential oversubscription.

| **KEY_PERF_COUNT**
| Parameter values: YES/NO
| Default value: NO
| Description:
|   A general option also supported by other plugins. It enables gathering performance counters.

| **KEY_CPU_THREADS_NUM**
| Parameter values: positive integer values
| Default value: 0
| Description:
|    Specifies the number of threads that CPU plugin should use for inference. Zero (default) means using all (logical) cores.

| **KEY_CPU_BIND_THREAD**
| Parameter values: YES/NUMA/NO
| Default value: YES
| Description:
|    Binds inference threads to CPU cores. 'YES' (default) binds option map threads to cores - this works best for static/synthetic scenarios like benchmarks. The 'NUMA' binding is more relaxed, binding inference threads only to NUMA nodes, leaving further scheduling to specific cores to the OS. This option might perform better in the real-life/contended scenarios. Note that for the latency-oriented cases (the number of streams is less or equal to the number of NUMA nodes, see below) both YES and NUMA options limit the number of inference threads to the number of hardware cores (ignoring hyper-threading) on the multi-socket machines.

| **KEY_CPU_THROUGHPUT_STREAMS**
| Parameter values: KEY_CPU_THROUGHPUT_NUMA, KEY_CPU_THROUGHPUT_AUTO, or positive integer values
| Default value: 1
| Description:
|    Specifies the number of CPU "execution" streams for the throughput mode, the upper bound for the number of inference requests that can be executed simultaneously. All available CPU cores are evenly distributed between the streams. The default value is 1, which implies latency-oriented behavior for a single NUMA-node machine, with all available cores processing requests one by one. On a multi-socket machine (multiple NUMA nodes), the best latency numbers are usually achieved with a number of streams matching the number of NUMA-nodes. 
|    KEY_CPU_THROUGHPUT_NUMA creates as many streams as needed to accommodate NUMA and avoid associated penalties.
|    KEY_CPU_THROUGHPUT_AUTO creates the bare minimum of streams to improve the performance; this is the most portable option if you don't know how many cores your target machine has (and what the optimal number of streams would be). Note that your application should provide enough parallel slack (for example, run many inference requests) to leverage the throughput mode.
|    A non-negative integer value creates the requested number of streams. If the number of streams is 0, no internal streams are created and user threads are interpreted as stream master threads.

| **KEY_ENFORCE_BF16**
| Parameter values: YES/NO
| Default value: YES
| Description:
|    The setting allowing to execute in the bfloat16 precision whenever possible. It tells the plugin to downscale precision where it sees performance benefits from bfloat16 execution. However, this approach does not guarantee accuracy of the network and you need to verify the accuracy separately, based on performance and accuracy results. It should be your decision whether to use this option or not. 

@endsphinxdirective


@sphinxdirective
.. note::

   To disable all internal threading, use the following set of configuration parameters: ``KEY_CPU_THROUGHPUT_STREAMS=0``, ``KEY_CPU_THREADS_NUM=1``, ``KEY_CPU_BIND_THREAD=NO``.
@endsphinxdirective



## See Also
* [Supported Devices](Supported_Devices.md)

[mkldnn_group_conv]: ../img/mkldnn_group_conv.png
[mkldnn_conv_sum]: ../img/mkldnn_conv_sum.png
[mkldnn_conv_sum_result]: ../img/mkldnn_conv_sum_result.png
[conv_simple_01]: ../img/conv_simple_01.png
[pooling_fakequant_01]: ../img/pooling_fakequant_01.png
[fullyconnected_activation_01]: ../img/fullyconnected_activation_01.png
[conv_depth_01]: ../img/conv_depth_01.png
[group_convolutions_01]: ../img/group_convolutions_01.png
[conv_sum_relu_01]: ../img/conv_sum_relu_01.png
