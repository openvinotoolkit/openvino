# Running on Multiple Devices Simultaneously {#openvino_docs_OV_UG_Running_on_multiple_devices}

@sphinxdirective

To run inference on multiple devices, you can choose either of the following ways:
- Use the :ref:`CUMULATIVE_THROUGHPUT performance option <cumulative throughput>` of the :ref:`Automatic Device Selection mode <openvino_docs_OV_UG_supported_plugins_AUTO>`. This way, you can use all available devices in the system without the need to specify them. 
- Use the Multi-Device execution mode. This page will explain how it works and how to use it.

@endsphinxdirective

## How MULTI Works

The Multi-Device execution mode, or MULTI for short, acts as a "virtual" or a "proxy" device, which does not bind to a specific type of hardware. Instead, it assigns available computing devices to particular inference requests, which are then executed in parallel. 

The potential gains from using Multi-Device execution are:
* improved throughput from using multiple devices at once,
* increase in performance stability due to multiple devices sharing inference workload.

Importantly, the Multi-Device mode does not change the application logic, so it does not require you to explicitly compile the model on every device or create and balance inference requests. It appears to use a typical device but internally handles the actual hardware.

Note that the performance increase in this mode comes from utilizing multiple devices at once. This means that you need to provide the devices with enough inference requests to keep them busy, otherwise you will not benefit much from using MULTI.


## Using the Multi-Device Mode 

Following the OpenVINO™ naming convention, the Multi-Device mode is assigned the label of “MULTI.” The only configuration option available for it is a prioritized list of devices to use:

@sphinxdirective

+---------------------------+---------------------------------+------------------------------------------------------------+
| Property                  | Property values                 | Description                                                |
+===========================+=================================+============================================================+
| <device list>             | | MULTI: <device names>         | | Specifies the devices available for selection.           |
|                           | | comma-separated, no spaces    | | The device sequence will be taken as priority            |
+---------------------------+---------------------------------+ | from high to low.                                        |
| ov::device::priorities    | | device names                  | | Priorities can be set directly as a string.              |
|                           | | comma-separated, no spaces    |                                                            |
+---------------------------+---------------------------------+------------------------------------------------------------+

@endsphinxdirective

Specifying the device list explicitly is required by MULTI, as it defines the devices available for inference and sets their priorities.  Importantly, the list may also specify the number of requests for MULTI to keep for each device, as described below.

Note that OpenVINO™ Runtime enables you to use “GPU” as an alias for “GPU.0” in function calls. More details on enumerating devices can be found in [Working with devices](supported_plugins/Device_Plugins.md).

The following commands are accepted by the API:

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/MULTI0.cpp
       :language: cpp
       :fragment: [part0]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_multi.py
       :language: python
       :fragment: [MULTI_0]

@endsphinxdirective

Notice that MULTI allows you to **change device priorities on the fly**. You can alter the order, exclude a device, and bring an excluded device back. Still, it does not allow adding new devices.

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/MULTI1.cpp
       :language: cpp
       :fragment: [part1]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_multi.py
       :language: python
       :fragment: [MULTI_1]

@endsphinxdirective



One more thing you can define is the **number of requests to allocate for each device**. You can do it simply by adding the number to each device in parentheses, like this: `"MULTI:CPU(2),GPU(2)"`. However, this method is not recommended as it is not performance-portable. The suggested approach is to configure individual devices and query the resulting number of requests to be used at the application level, as described in [Configuring Individual Devices and Creating MULTI On Top](#configuring-the-individual-devices-and-creating-the-multi-device-on-top).

To check what devices are present in the system, you can use the Device API. For information on how to do it, check [Query device properties and configuration](supported_plugins/config_properties.md).


### Configuring Individual Devices and Creating the Multi-Device On Top
As mentioned previously, executing inference with MULTI may be set up by configuring individual devices before creating the "MULTI" device on top. It may be considered for performance reasons.

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/MULTI4.cpp
       :language: cpp
       :fragment: [part4]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_multi.py
       :language: python
       :fragment: [MULTI_4]

@endsphinxdirective

Alternatively, you can combine all the individual device settings into a single config file and load it for MULTI to parse. See the code example in the next section.



### Querying the Optimal Number of Inference Requests
When using MULTI, you don't need to sum over included devices yourself, you can query the optimal number of requests directly, 
using the [configure devices](supported_plugins/config_properties.md) property: 

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/MULTI5.cpp
       :language: cpp
       :fragment: [part5]

@endsphinxdirective



## Using the Multi-Device with OpenVINO Samples and Benchmarking Performance

To see how the Multi-Device execution is used in practice and test its performance, take a look at OpenVINO's Benchmark Application which presents the optimal performance of the plugin without the need for additional settings, like the number of requests or CPU threads. 
Here is an example command to evaluate performance of HDDL+GPU: 

```sh
./benchmark_app –d MULTI:HDDL,GPU –m <model> -i <input> -niter 1000
```

For more information, refer to the [C++](../../samples/cpp/benchmark_app/README.md) or [Python](../../tools/benchmark_tool/README.md) version instructions.	

@sphinxdirective
.. note::

   You can keep using the FP16 IR without converting it to FP32, even if some of the listed devices do not support it. The conversion will be done automatically for you.

   No demos are yet fully optimized for MULTI, by means of supporting the ov::optimal_number_of_infer_requests property, using the GPU streams/throttling, and so on.
@endsphinxdirective


## Performance Considerations for the Multi-Device Execution
For best performance when using the MULTI execution mode you should consider a few recommendations:
- MULTI usually performs best when the fastest device is specified first in the device candidate list. 
This is particularly important when the request-level parallelism is not sufficient 
(e.g. the number of requests is not enough to saturate all devices).
- Just like with any throughput-oriented execution mode, it is highly recommended to query the optimal number of inference requests 
directly from the instance of the `ov:compiled_model`. Refer to the code of the previously mentioned `benchmark_app` for more details.    
- Execution on certain device combinations, for example CPU+GPU, performs better with certain knobs. Refer to the `benchmark_app` code for details. One specific example is disabling GPU driver polling, which in turn requires multiple GPU streams to balance out slower 
communication of inference completion from the device to the host.
- The MULTI logic always attempts to save on copying data between device-agnostic and user-facing inference requests, 
and device-specific 'worker' requests that are being actually scheduled behind the scene. 
To facilitate the copy savings, it is recommended to run the requests in the order in which they were created.
- While performance of accelerators combines well with MULTI, the CPU+GPU execution may introduce certain performance issues. It is due to the devices sharing some resources, like power or bandwidth. Enabling the GPU throttling hint, which saves a CPU thread for CPU inference, is an example of a recommended solution addressing this issue.



## See Also

- [Supported Devices](supported_plugins/Supported_Devices.md)
- [Automatic Device Selection](./auto_device_selection.md)

@sphinxdirective
.. raw:: html

    <iframe allowfullscreen mozallowfullscreen msallowfullscreen oallowfullscreen webkitallowfullscreen width="560" height="315" src="https://www.youtube.com/embed/xbORYFEmrqU" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

@endsphinxdirective

> **NOTE**: This video is currently available only for C++, but many of the same concepts apply to Python.
