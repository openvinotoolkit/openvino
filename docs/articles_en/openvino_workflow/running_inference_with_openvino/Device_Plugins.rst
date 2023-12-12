.. {#openvino_docs_OV_UG_Working_with_devices}

Inference Device Support
========================


.. meta::
   :description: The list of types of devices and corresponding plugins which 
                 are compatible with OpenVINO Runtime and support inference 
                 of deep learning models.


.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_OV_UG_query_api
   openvino_docs_OV_UG_supported_plugins_CPU
   openvino_docs_OV_UG_supported_plugins_GPU
   openvino_docs_OV_UG_supported_plugins_NPU
   openvino_docs_OV_UG_supported_plugins_GNA


OpenVINO™ Runtime can infer deep learning models using the following device types:

* :doc:`CPU <openvino_docs_OV_UG_supported_plugins_CPU>`
* :doc:`GPU <openvino_docs_OV_UG_supported_plugins_GPU>`
* :doc:`GNA <openvino_docs_OV_UG_supported_plugins_GNA>`
* :doc:`Arm® CPU <openvino_docs_OV_UG_supported_plugins_CPU>`

For a more detailed list of hardware, see :doc:`Supported Devices <openvino_docs_OV_UG_supported_plugins_Supported_Devices>`.

.. _devicesupport-feature-support-matrix:



Feature Support Matrix
#######################################

The table below demonstrates support of key features by OpenVINO device plugins.

========================================================================================= ============================ =============== ===============
 Capability                                                                                CPU                         GPU             GNA            
========================================================================================= ============================ =============== ===============
 :doc:`Heterogeneous execution <openvino_docs_OV_UG_Hetero_execution>`                     Yes                         Yes             No            
 :doc:`Multi-device execution <openvino_docs_OV_UG_Running_on_multiple_devices>`           Yes                         Yes             Partial       
 :doc:`Automatic batching <openvino_docs_OV_UG_Automatic_Batching>`                        No                          Yes             No            
 :doc:`Multi-stream execution <openvino_docs_deployment_optimization_guide_tput>`          Yes (Intel® x86-64 only)    Yes             No             
 :doc:`Models caching <openvino_docs_OV_UG_Model_caching_overview>`                        Yes                         Partial         Yes            
 :doc:`Dynamic shapes <openvino_docs_OV_UG_DynamicShapes>`                                 Yes                         Partial         No             
 :doc:`Import/Export <openvino_ecosystem>`                                                 Yes                         No              Yes            
 :doc:`Preprocessing acceleration <openvino_docs_OV_UG_Preprocessing_Overview>`            Yes                         Yes             No            
 :doc:`Stateful models <openvino_docs_OV_UG_model_state_intro>`                            Yes                         No              Yes            
 :doc:`Extensibility <openvino_docs_Extensibility_UG_Intro>`                               Yes                         Yes             No            
========================================================================================= ============================ =============== ===============

For more details on plugin-specific feature limitations, see the corresponding plugin pages.

Enumerating Available Devices
#######################################

The OpenVINO Runtime API features dedicated methods of enumerating devices and their capabilities. See the :doc:`Hello Query Device C++ Sample <openvino_inference_engine_samples_hello_query_device_README>`. This is an example output from the sample (truncated to device names only):

.. code-block:: sh

   ./hello_query_device
   Available devices:
       Device: CPU
   ...
       Device: GPU.0
   ...
       Device: GPU.1
   ...
       Device: GNA


A simple programmatic way to enumerate the devices and use with the multi-device is as follows:

.. tab-set::

   .. tab-item:: C++
      :sync: cpp
   
      .. doxygensnippet:: docs/snippets/MULTI2.cpp
         :language: cpp
         :fragment: [part2]



Beyond the typical "CPU", "GPU", and so on, when multiple instances of a device are available, the names are more qualified. 
For example, this is how two GPUs can be listed (iGPU is always GPU.0):

.. code-block:: sh

   ...
       Device: GPU.0
   ...
       Device: GPU.1


So, the explicit configuration to use both would be "MULTI:GPU.1,GPU.0". Accordingly, the code that loops over all available devices of the "GPU" type only is as follows:

.. tab-set::

   .. tab-item:: C++
      :sync: cpp
   
      .. doxygensnippet:: docs/snippets/MULTI3.cpp
         :language: cpp
         :fragment: [part3]





