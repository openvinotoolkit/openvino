Multi-device execution
======================


.. meta::
   :description: The Multi-Device execution mode in OpenVINO Runtime assigns
                 multiple available computing devices to particular inference
                 requests to execute in parallel.

.. danger::

   The Multi-device execution mode described here has been **deprecated**.

   It's functionality is now fully covered by the :ref:`CUMULATIVE_THROUGHPUT <cumulative throughput>`
   option of the :doc:`Automatic Device Selection <../../openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection>` mode.
   This way, all available devices in the system can be used without the need to specify them.

How MULTI Works
####################

The Multi-Device execution mode, or MULTI for short, acts as a "virtual" or a "proxy" device, which does not bind to a specific type of hardware. Instead, it assigns available computing devices to particular inference requests, which are then executed in parallel.

The potential gains from using Multi-Device execution are:

* improved throughput from using multiple devices at once,
* increase in performance stability due to multiple devices sharing inference workload.

Importantly, the Multi-Device mode does not change the application logic, so it does not require you to explicitly compile the model on every device or create and balance inference requests. It appears to use a typical device but internally handles the actual hardware.

Note that the performance increase in this mode comes from utilizing multiple devices at once. This means that you need to provide the devices with enough inference requests to keep them busy, otherwise you will not benefit much from using MULTI.


Using the Multi-Device Mode
###########################

Following the OpenVINO™ naming convention, the Multi-Device mode is assigned the label of “MULTI.” The only configuration option available for it is a prioritized list of devices to use:


+----------------------------+---------------------------------+------------------------------------------------------------+
| Property                   | Property values                 | Description                                                |
+============================+=================================+============================================================+
| <device list>              | | MULTI: <device names>         | | Specifies the devices available for selection.           |
|                            | | comma-separated, no spaces    | | The device sequence will be taken as priority            |
+----------------------------+---------------------------------+ | from high to low.                                        |
| ``ov::device::priorities`` | | device names                  | | Priorities can be set directly as a string.              |
|                            | | comma-separated, no spaces    |                                                            |
+----------------------------+---------------------------------+------------------------------------------------------------+


Specifying the device list explicitly is required by MULTI, as it defines the devices available for inference and sets their priorities.

Note that OpenVINO™ Runtime enables you to use “GPU” as an alias for “GPU.0” in function calls.
More details on enumerating devices can be found in :doc:`Inference Devices and Modes <../../openvino-workflow/running-inference/inference-devices-and-modes>`.

The following commands are accepted by the API:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_multi.py
         :language: python
         :fragment: [MULTI_0]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/MULTI0.cpp
         :language: cpp
         :fragment: [part0]


To check what devices are present in the system, you can use the Device API. For information on how to do it, check :doc:`Query device properties and configuration <../../openvino-workflow/running-inference/inference-devices-and-modes/query-device-properties>`.


Configuring Individual Devices and Creating the Multi-Device On Top
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

As mentioned previously, executing inference with MULTI may be set up by configuring individual devices before creating the "MULTI" device on top. It may be considered for performance reasons.

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_multi.py
         :language: python
         :fragment: [MULTI_4]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/MULTI4.cpp
         :language: cpp
         :fragment: [part4]


Alternatively, you can combine all the individual device settings into a single config file and load it for MULTI to parse. See the code example in the next section.

Querying the Optimal Number of Inference Requests
+++++++++++++++++++++++++++++++++++++++++++++++++

When using MULTI, you don't need to sum over included devices yourself, you can query the optimal number of requests directly,
using the :doc:`configure devices <../../openvino-workflow/running-inference/inference-devices-and-modes/query-device-properties>` property:

.. tab-set::

   .. tab-item:: C++

       .. doxygensnippet:: docs/articles_en/assets/snippets/MULTI5.cpp
          :language: cpp
          :fragment: [part5]


Using the Multi-Device with OpenVINO Samples and Benchmarking Performance
#########################################################################

To see how the Multi-Device execution is used in practice and test its performance, take a look at OpenVINO's Benchmark Application which presents the optimal performance of the plugin without the need for additional settings, like the number of requests or CPU threads.
Here is an example command to evaluate performance of CPU + GPU:

.. code-block:: sh

   ./benchmark_app –d MULTI:CPU,GPU –m <model> -i <input> -niter 1000


For more information, refer to the :doc:`Benchmark Tool <../../../learn-openvino/openvino-samples/benchmark-tool>` article.


.. note::

   You can keep using the FP16 IR without converting it to FP32, even if some of the listed devices do not support it. The conversion will be done automatically for you.

   No demos are yet fully optimized for MULTI, by means of supporting the ``ov::optimal_number_of_infer_requests`` property, using the GPU streams/throttling, and so on.


Performance Considerations for the Multi-Device Execution
#########################################################

For best performance when using the MULTI execution mode you should consider a few recommendations:

- MULTI usually performs best when the fastest device is specified first in the device candidate list. This is particularly important when the request-level parallelism is not sufficient (e.g. the number of requests is not enough to saturate all devices).
- Just like with any throughput-oriented execution mode, it is highly recommended to query the optimal number of inference requests directly from the instance of the ``ov:compiled_model``. Refer to the code of the previously mentioned ``benchmark_app`` for more details.
- Execution on certain device combinations, for example CPU+GPU, performs better with certain knobs. Refer to the ``benchmark_app`` code for details. One specific example is disabling GPU driver polling, which in turn requires multiple GPU streams to balance out slower communication of inference completion from the device to the host.
- The MULTI logic always attempts to save on copying data between device-agnostic and user-facing inference requests, and device-specific 'worker' requests that are being actually scheduled behind the scene. To facilitate the copy savings, it is recommended to run the requests in the order in which they were created.
- While performance of accelerators combines well with MULTI, the CPU+GPU execution may introduce certain performance issues. It is due to the devices sharing some resources, like power or bandwidth. Enabling the GPU throttling hint, which saves a CPU thread for CPU inference, is an example of a recommended solution addressing this issue.


Additional Resources
####################

- :doc:`Inference Devices and Modes <../../openvino-workflow/running-inference/inference-devices-and-modes>`
- :doc:`Automatic Device Selection <../../openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection>`


