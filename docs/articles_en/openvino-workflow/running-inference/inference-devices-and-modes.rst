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


The OpenVINO™ Runtime offers several inference modes to optimize hardware usage.
You can run inference on a single device or use automated modes that manage multiple devices:

| **single-device inference**
|    This mode runs all inference on one selected device. The OpenVINO Runtime includes 
     built-in plugins that support the following devices:
|    :doc:`CPU <inference-devices-and-modes/cpu-device>`
|    :doc:`GPU <inference-devices-and-modes/gpu-device>`
|    :doc:`NPU <inference-devices-and-modes/npu-device>`

| **automated inference modes**
|    These modes automate device selection and workload distribution, potentially increasing 
     performance and portability:
|    :doc:`Automatic Device Selection (AUTO) <inference-devices-and-modes/auto-device-selection>`
|    :doc:`Heterogeneous Execution (HETERO) <inference-devices-and-modes/hetero-execution>`  across different device types
|    :doc:`Automatic Batching Execution (Auto-batching) <inference-devices-and-modes/automatic-batching>`: automatically groups inference requests to improve throughput

Learn how to configure devices in the :doc:`Query device properties <inference-devices-and-modes/query-device-properties>` article.

Enumerating Available Devices
#######################################

The OpenVINO Runtime API provides methods to list available devices and their details.
When there are multiple instances of a device, they get specific names like GPU.0 for iGPU.
Here is an example of the output with device names, including two GPUs:

.. code-block:: sh

   ./hello_query_device
   Available devices:
       Device: CPU
   ...
       Device: GPU.0
   ...
       Device: GPU.1


See the :doc:`Hello Query Device Sample <../../get-started/learn-openvino/openvino-samples/hello-query-device>`
for more details.

Below is an example showing how to list available devices and use them with multi-device mode:

.. tab-set::

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/MULTI2.cpp
         :language: cpp
         :fragment: [part2]

If you have two GPU devices, you can specify them explicitly as “MULTI:GPU.1,GPU.0”. 
Here is how to list and use all available GPU devices:

.. tab-set::

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/MULTI3.cpp
         :language: cpp
         :fragment: [part3]

Additional Resources
####################

* `OpenVINO™ Runtime API Tutorial <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/openvino-api>`__
* `AUTO Device Tutorial <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/auto-device>`__
* `GPU Device Tutorial <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/gpu-device>`__
* `NPU Device Tutorial <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/hello-npu>`__