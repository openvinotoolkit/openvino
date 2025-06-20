Inference Devices and Modes
===============================================================================================


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


Inference modes in OpenVINO enable the best hardware utilization under different conditions:

- **single-device inference**

  | Select just one device for the entire inference workload.
    The mode supports a range of processors by means of the following plugins:

  * :doc:`CPU <inference-devices-and-modes/cpu-device>`
  * :doc:`GPU <inference-devices-and-modes/gpu-device>`
  * :doc:`NPU <inference-devices-and-modes/npu-device>`

- **automated inference modes**

  | Introduce automation in selecting devices for better inference.
    They may potentially increase performance and portability of your deployed solution.
    The automated modes are:

  * :doc:`Automatic Device Selection (AUTO) <inference-devices-and-modes/auto-device-selection>`
  * :doc:`Heterogeneous Execution (HETERO) <inference-devices-and-modes/hetero-execution>`
  * :doc:`Automatic Batching Execution (Auto-batching) <inference-devices-and-modes/automatic-batching>`

To learn how to change the device configuration, read the
:doc:`Query device properties article <inference-devices-and-modes/query-device-properties>`.

Enumerating Available Devices
###############################################################################################

The OpenVINO Runtime API features dedicated methods of enumerating devices and their
capabilities. Note that beyond the typical “CPU” or “GPU” device names, more qualified names
are used when multiple instances of a device are available (iGPU is always GPU.0).
Also, the output you receive may be truncated to device names only:

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

Below is an example of how to enumerate the devices and use them with the multi-device mode:

.. tab-set::

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/MULTI2.cpp
         :language: cpp
         :fragment: [part2]

With two GPU devices used in one setup, the explicit configuration would be
“MULTI:GPU.1,GPU.0”. Accordingly, the code that loops over all available devices of
the “GPU” type only is as follows:

.. tab-set::

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/MULTI3.cpp
         :language: cpp
         :fragment: [part3]


Additional Resources
###############################################################################################

* `OpenVINO™ Runtime API Tutorial <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/openvino-api>`__
* `AUTO Device Tutorial <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/auto-device>`__
* `GPU Device Tutorial <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/gpu-device>`__
* `NPU Device Tutorial <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/hello-npu>`__
