.. {#openvino_docs_Runtime_Inference_Modes_Overview}

Inference Devices and Modes
============================


.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_OV_UG_supported_plugins_CPU
   openvino_docs_OV_UG_supported_plugins_GPU
   openvino_docs_OV_UG_supported_plugins_NPU
   openvino_docs_OV_UG_supported_plugins_AUTO
   openvino_docs_OV_UG_Running_on_multiple_devices
   openvino_docs_OV_UG_Hetero_execution
   openvino_docs_OV_UG_Automatic_Batching
   openvino_docs_OV_UG_query_api


The OpenVINO runtime offers multiple inference modes to enable the best hardware utilization under
different conditions:

| **single-device inference**
|    Define just one device responsible for the entire inference workload. It supports a range of
     processors by means of the following plugins embedded in the Runtime library:
|    - :doc:`CPU <openvino_docs_OV_UG_supported_plugins_CPU>`
|    - :doc:`GPU <openvino_docs_OV_UG_supported_plugins_GPU>`
|    - :doc:`NPU <openvino_docs_OV_UG_supported_plugins_NPU>`

| **automated inference modes**
|    Assume certain level of automation in selecting devices for inference. They may potentially
     increase your deployed solution's performance and portability. The automated modes are:
|    - :doc:`Automatic Device Selection (AUTO) <openvino_docs_OV_UG_supported_plugins_AUTO>`
|    - :doc:`Multi-Device Execution (MULTI) <openvino_docs_OV_UG_Running_on_multiple_devices>`
|    - :doc:`Heterogeneous Execution (HETERO) <openvino_docs_OV_UG_Hetero_execution>`
|    - :doc:`Automatic Batching Execution (Auto-batching) <openvino_docs_OV_UG_Automatic_Batching>`



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


You may see how to obtain this information in the :doc:`Hello Query Device Sample <openvino_sample_hello_query_device>`.
Here is an example of a simple programmatic way to enumerate the devices and use them with the
multi-device mode:

.. tab-set::

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/MULTI2.cpp
         :language: cpp
         :fragment: [part2]

With two GPU devices used in one setup, the explicit configuration would be "MULTI:GPU.1,GPU.0".
Accordingly, the code that loops over all available devices of the "GPU" type only is as follows:

.. tab-set::

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/MULTI3.cpp
         :language: cpp
         :fragment: [part3]


