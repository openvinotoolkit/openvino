Inference Devices and Modes
============================


.. toctree::
   :maxdepth: 1
   :hidden:

   inference-devices-and-modes/cpu-device
   inference-devices-and-modes/gpu-device
   inference-devices-and-modes/npu-device
   inference-devices-and-modes/auto-device-selection
   inference-devices-and-modes/hetero-execution
   inference-devices-and-modes/automatic-batching
   inference-devices-and-modes/query-device-properties


The OpenVINO runtime offers multiple inference modes to enable the best hardware utilization under
different conditions:

| **single-device inference**
|    Define just one device responsible for the entire inference workload. It supports a range of
     processors by means of the following plugins embedded in the Runtime library:
|    :doc:`CPU <inference-devices-and-modes/cpu-device>`
|    :doc:`GPU <inference-devices-and-modes/gpu-device>`
|    :doc:`NPU <inference-devices-and-modes/npu-device>`

| **automated inference modes**
|    Assume certain level of automation in selecting devices for inference. They may potentially
     increase your deployed solution's performance and portability. The automated modes are:
|    :doc:`Automatic Device Selection (AUTO) <inference-devices-and-modes/auto-device-selection>`
|    :doc:`Heterogeneous Execution (HETERO) <inference-devices-and-modes/hetero-execution>`
|    :doc:`Automatic Batching Execution (Auto-batching) <inference-devices-and-modes/automatic-batching>`
|    :doc:`[DEPRECATED] Multi-Device Execution (MULTI) <../../documentation/legacy-features/multi-device>`

To learn how to change the device configuration, read the :doc:`Query device properties article <inference-devices-and-modes/query-device-properties>`.

Enumerating Available Devices
#######################################

The OpenVINO Runtime API features dedicated methods of enumerating devices and their capabilities.
Note that beyond the typical "CPU" or "GPU" device names, more qualified names are used when multiple
instances of a device are available (iGPU is always GPU.0).
The output you receive may look like this (truncated to device names only, two GPUs are listed
as an example):

.. code-block:: sh

   ./hello_query_device
   Available devices:
       Device: CPU
   ...
       Device: GPU.0
   ...
       Device: GPU.1


You may see how to obtain this information in the :doc:`Hello Query Device Sample <../../learn-openvino/openvino-samples/hello-query-device>`.
Here is an example of a simple programmatic way to enumerate the devices and use them with the
multi-device mode:

.. tab-set::

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/MULTI2.cpp
         :language: cpp
         :fragment: [part2]

With two GPU devices used in one setup, the explicit configuration would be "MULTI:GPU.1,GPU.0".
Accordingly, the code that loops over all available devices of the "GPU" type only is as follows:

.. tab-set::

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/MULTI3.cpp
         :language: cpp
         :fragment: [part3]

Additional Resources
####################

* `OpenVINO™ Runtime API Tutorial <./../../notebooks/openvino-api-with-output.html>`__
* `AUTO Device Tutorial <./../../notebooks/auto-device-with-output.html>`__
* `GPU Device Tutorial <./../../notebooks/gpu-device-with-output.html>`__
* `NPU Device Tutorial <./../../notebooks/hello-npu-with-output.html>`__