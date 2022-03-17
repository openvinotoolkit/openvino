# Automatic device selection {#openvino_docs_IE_DG_supported_plugins_AUTO}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   Debugging Auto-Device Plugin <openvino_docs_IE_DG_supported_plugins_AUTO_debugging>

@endsphinxdirective

The Auto-Device plugin, or AUTO, is a virtual device which automatically selects the processing unit to use for inference with OpenVINO™. It chooses from a list of available devices defined by the user and aims at finding the most suitable hardware for the given model. The best device is chosen using the following logic: 

1. Check which supported devices are available. 
2. Check the precision of the input model (for detailed information on precisions read more on the `ov::device::capabilities`) 
3. From the priority list, including all available devices if the user does not alter the defaults, select the first device capable of supporting the given precision, as presented in the table below.
4. If the network’s precision is FP32 but there is no device capable of supporting it, offload the network to a device supporting FP16. 

@sphinxdirective
+----------+------------------------------------------------------+-------------------------------------+
| Choice   | | Supported                                          | | Supported                         |
| Priority | | Device                                             | | model precision                   |
+==========+======================================================+=====================================+
| 1        | | dGPU                                               | FP32, FP16, INT8, BIN               |
|          | | (e.g. Intel® Iris® Xe MAX)                         |                                     |
+----------+------------------------------------------------------+-------------------------------------+
| 2        | | iGPU                                               | FP32, FP16, BIN,                    |
|          | | (e.g. Intel® UHD Graphics 620 (iGPU))              |                                     |
+----------+------------------------------------------------------+-------------------------------------+
| 3        | | Intel® Movidius™ Myriad™ X VPU                     | FP16                                |
|          | | (e.g. Intel® Neural Compute Stick 2 (Intel® NCS2)) |                                     |
+----------+------------------------------------------------------+-------------------------------------+
| 4        | | Intel® CPU                                         | FP32, FP16, INT8, BIN               |
|          | | (e.g. Intel® Core™ i7-1165G7)                      |                                     |
+----------+------------------------------------------------------+-------------------------------------+
@endsphinxdirective

To put it simply, when loading the network to the first device on the list fails, AUTO will try to load it to the next device in line, until one of them succeeds. For example: 
If you have dGPU in your system, it will be selected for most jobs (first on the priority list and supports multiple precisions). But if you want to run a WINOGRAD-enabled IR, your CPU will be selected (WINOGRAD optimization is not supported by dGPU). If you have Myriad and IA CPU in your system, Myriad will be selected for FP16 models, but IA CPU will be chosen for FP32 ones.  

What is important, **AUTO always starts inference with the CPU**. CPU provides very low latency and can start inference with no additional delays. While it performs inference, the Auto-Device plugin continues to load the model to the device best suited for the purpose and transfers the task to it when ready. This way, the devices which are much slower in loading the network, GPU being the best example, do not impede inference at its initial stages. 

This mechanism can be easily observed in our Benchmark Application sample ([see here](#Benchmark App Info)), showing how the first-inference latency (the time it takes to load the network and perform the first inference) is reduced when using AUTO. For example: 

@sphinxdirective
.. code-block:: sh

   ./benchmark_app -m ../public/alexnet/FP32/alexnet.xml -d GPU -niter 128
@endsphinxdirective 

first-inference latency: **2594.29 ms + 9.21 ms** 

@sphinxdirective
.. code-block:: sh

   ./benchmark_app -m ../public/alexnet/FP32/alexnet.xml -d AUTO:CPU,GPU -niter 128
@endsphinxdirective 

first-inference latency: **173.13 ms + 13.20 ms**

@sphinxdirective
.. note::
   The realtime performance will be closer to the best suited device the longer the process runs.
@endsphinxdirective

## Using the Auto-Device Plugin 

Inference with AUTO is configured similarly to other plugins: first you configure devices, then load a network to the plugin, and finally, execute inference. 

Following the OpenVINO™ naming convention, the Auto-Device plugin is assigned the label of “AUTO.” It may be defined with no additional parameters, resulting in defaults being used, or configured further with the following setup options: 

@sphinxdirective
+---------------------------+-----------------------------------------------+-----------------------------------------------------------+
| Property                  | Property values                               | Description                                               |
+===========================+===============================================+===========================================================+
| <device candidate list>   | | AUTO: <device names>                        | | Lists the devices available for selection.              |
|                           | | comma-separated, no spaces                  | | The device sequence will be taken as priority           |
|                           | |                                             | | from high to low.                                       |
|                           | |                                             | | If not specified, “AUTO” will be used as default        |
|                           | |                                             | | and all devices will be included.                       |
+---------------------------+-----------------------------------------------+-----------------------------------------------------------+
| ov::device:priorities     | | device names                                | | Specifies the devices for Auto-Device plugin to select. |
|                           | | comma-separated, no spaces                  | | The device sequence will be taken as priority           |
|                           | |                                             | | from high to low.                                       |
|                           | |                                             | | This configuration is optional.                         |
+---------------------------+-----------------------------------------------+-----------------------------------------------------------+
| ov::hint::performance_mode| | ov::hint::PerformanceMode::LATENCY          | | Specifies the performance mode preferred                |
|                           | | ov::hint::PerformanceMode::THROUGHPUT       | | by the application.                                     |
+---------------------------+-----------------------------------------------+-----------------------------------------------------------+
| ov::hint::model_priority  | | ov::hint::Priority::HIGH                    | | Indicates the priority for a network.                   |
|                           | | ov::hint::Priority::MEDIUM                  | | Importantly!                                            |
|                           | | ov::hint::Priority::LOW                     | | This property is still not fully supported              |
+---------------------------+-----------------------------------------------+-----------------------------------------------------------+
@endsphinxdirective

@sphinxdirective
.. dropdown:: Click for information on Legacy APIs 

   For legacy APIs like LoadNetwork/SetConfig/GetConfig/GetMetric:
   
   - replace {ov::device:priorities, "GPU,CPU"} with {"MULTI_DEVICE_PRIORITIES", "GPU,CPU"}
   - replace {ov::hint:model_priority, "LOW"} with {"MODEL_PRIORITY", "LOW"}
   - InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES is defined as same string "MULTI_DEVICE_PRIORITIES"
   - InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY is defined as same string "MODEL_PRIORITY"
   - InferenceEngine::PluginConfigParams::MODEL_PRIORITY_LOW is defined as same string "LOW"
@endsphinxdirective

### Device candidate list
The device candidate list allows users to customize the priority and limit the choice of devices available to the AUTO plugin. If not specified, the plugin assumes all the devices present in the system can be used. Note, that OpenVINO™ Runtime lets you use “GPU” as an alias for “GPU.0” in function calls. 
The following commands are accepted by the API: 

@sphinxdirective

.. tab:: OpenVINO Runtime API C++

    .. doxygensnippet:: docs/snippets/AUTO0.cpp
       :language: cpp
       :fragment: [part0]

.. tab:: Inference Engine API C++

    .. doxygensnippet:: docs/snippets/AUTO1.cpp
       :language: cpp
       :fragment: [part1]

.. tab:: OpenVINO Runtime API Python

    .. doxygensnippet:: docs/snippets/ov_auto.py
       :language: python
       :fragment: [part0]

.. tab:: Inference Engine API Python

    .. doxygensnippet:: docs/snippets/ov_auto.py
       :language: python
       :fragment: [part1]

@endsphinxdirective

To check what devices are present in the system, you can use Device API. For information on how to do it, check [Query device properties and configuration](supported_plugins/config_properties.md)

For OpenVINO Runtime API C++
@sphinxdirective
.. code-block:: sh

   ov::runtime::Core::get_available_devices() (see Hello Query Device C++ Sample)
@endsphinxdirective

For OpenVINO Runtime API Python
@sphinxdirective
.. code-block:: sh

   openvino.runtime.Core.available_devices (see Hello Query Device Python Sample)
@endsphinxdirective


### Performance Hints
The `ov::hint::performance_mode` property enables you to specify a performance mode for the plugin to be more efficient for particular use cases.

#### ov::hint::PerformanceMode::THROUGHPUT
This mode prioritizes high throughput, balancing between latency and power. It is best suited for tasks involving multiple jobs, like inference of video feeds or large numbers of images.

#### ov::hint::PerformanceMode::LATENCY
This mode prioritizes low latency, providing short response time for each inference job. It performs best for tasks where inference is required for a single input image, like a medical analysis of an ultrasound scan image. It also fits the tasks of real-time or nearly real-time applications, such as an industrial robot's response to actions in its environment or obstacle avoidance for autonomous vehicles.
Note that currently the `ov::hint` property is supported by CPU and GPU devices only.

To enable performance hints for your application, use the following code: 
@sphinxdirective

.. tab:: OpenVINO Runtime API C++

    .. doxygensnippet:: docs/snippets/AUTO3.cpp
       :language: cpp
       :fragment: [part3]
 
.. tab:: OpenVINO Runtime API Python

    .. doxygensnippet:: docs/snippets/ov_auto.py
       :language: python
       :fragment: [part3]

@endsphinxdirective

### ov::hint::model_priority
The property enables you to control the priorities of networks in the Auto-Device plugin. A high-priority network will be loaded to a supported high-priority device. A lower-priority network will not be loaded to a device that is occupied by a higher-priority network.

@sphinxdirective

.. tab:: OpenVINO Runtime API C++

    .. doxygensnippet:: docs/snippets/AUTO4.cpp
       :language: cpp
       :fragment: [part4]
 
.. tab:: OpenVINO Runtime API Python

    .. doxygensnippet:: docs/snippets/ov_auto.py
       :language: python
       :fragment: [part4]

@endsphinxdirective

## Configuring Individual Devices and Creating the Auto-Device plugin on Top
Although the methods described above are currently the preferred way to execute inference with AUTO, the following steps can be also used as an alternative. It is currently available as a legacy feature and used if the device candidate list includes Myriad (devices uncapable of utilizing the Performance Hints option). 

@sphinxdirective

.. tab:: OpenVINO Runtime API C++

    .. doxygensnippet:: docs/snippets/AUTO5.cpp
       :language: cpp
       :fragment: [part5]
 
.. tab:: OpenVINO Runtime API Python

    .. doxygensnippet:: docs/snippets/ov_auto.py
       :language: python
       :fragment: [part5]

@endsphinxdirective

<a name="Benchmark App Info"></a>
## Using AUTO with OpenVINO™ Samples and the Benchmark App
To see how the Auto-Device plugin is used in practice and test its performance, take a look at OpenVINO™ samples. All samples supporting the "-d" command-line option (which stands for "device") will accept the plugin out-of-the-box. The Benchmark Application will be a perfect place to start – it presents the optimal performance of the plugin without the need for additional settings, like the number of requests or CPU threads. To evaluate the AUTO performance, you can use the following commands:

For unlimited device choice:
@sphinxdirective
.. code-block:: sh

   ./benchmark_app –d AUTO –m <model> -i <input> -niter 1000
@endsphinxdirective

For limited device choice:
@sphinxdirective
.. code-block:: sh

   ./benchmark_app –d AUTO:CPU,GPU,MYRIAD –m <model> -i <input> -niter 1000
@endsphinxdirective

For more information, refer to the [C++](../../samples/cpp/benchmark_app/README.md) or [Python](../../tools/benchmark_tool/README.md) version instructions.	

@sphinxdirective
.. note::

   The default CPU stream is 1 if using “-d AUTO”.

   You can use the FP16 IR to work with auto-device.

   No demos are yet fully optimized for AUTO, by means of selecting the most suitable device, using the GPU streams/throttling, and so on.
@endsphinxdirective
