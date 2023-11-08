# Supported Devices {#openvino_docs_OV_UG_supported_plugins_Supported_Devices}


@sphinxdirective

.. meta::
   :description: Check the list of officially supported models in Intel® 
                 Distribution of OpenVINO™ toolkit.


OpenVINO enables you to implement its inference capabilities in your own software,
utilizing various hardware. It currently supports the following processing units 
(for more details, see :doc:`system requirements <system_requirements>`):

* :doc:`CPU <openvino_docs_OV_UG_supported_plugins_CPU>`            
* :doc:`GPU <openvino_docs_OV_UG_supported_plugins_GPU>`            
* :doc:`GNA <openvino_docs_OV_UG_supported_plugins_GNA>`         
* :doc:`NPU <openvino_docs_OV_UG_supported_plugins_NPU>` 

.. note::

   GNA, currently available in the Intel® Distribution of OpenVINO™ toolkit,
   will be deprecated together with the hardware being discontinued 
   in future CPU solutions.   
   
   With OpenVINO™ 2023.0 release, support has been cancelled for:
   - Intel® Neural Compute Stick 2 powered by the Intel® Movidius™ Myriad™ X
   - Intel® Vision Accelerator Design with Intel® Movidius™
   
   To keep using the MYRIAD and HDDL plugins with your hardware, revert to the OpenVINO 2022.3 LTS release.


Beside running inference with a specific device, 
OpenVINO offers automated inference management with the following inference modes:

* :doc:`Automatic Device Selection <openvino_docs_OV_UG_supported_plugins_AUTO>` - automatically selects the best device 
  available for the given task. It offers many additional options and optimizations, including inference on 
  multiple devices at the same time.
* :doc:`Multi-device Inference <openvino_docs_OV_UG_Running_on_multiple_devices>` - executes inference on multiple devices. 
  Currently, this mode is considered a legacy solution. Using Automatic Device Selection is advised.
* :doc:`Heterogeneous Inference <openvino_docs_OV_UG_Hetero_execution>` - enables splitting inference among several devices 
  automatically, for example, if one device doesn’t support certain operations.


Devices similar to the ones used for benchmarking can be accessed using `Intel® DevCloud for the Edge <https://devcloud.intel.com/edge/>`__, 
a remote development environment with access to Intel® hardware and the latest versions of the Intel® Distribution 
of OpenVINO™ Toolkit. `Learn more <https://devcloud.intel.com/edge/get_started/devcloud/>`__ or `Register here <https://inteliot.force.com/DevcloudForEdge/s/>`__.


To learn more about each of the supported devices and modes, refer to the sections of:
* :doc:`Inference Device Support <openvino_docs_OV_UG_Working_with_devices>` 
* :doc:`Inference Modes <openvino_docs_Runtime_Inference_Modes_Overview>`

For setting up a relevant configuration, refer to the
:doc:`Integrate with Customer Application <openvino_docs_OV_UG_Integrate_OV_with_your_application>` 
topic (step 3 "Configure input and output").



@endsphinxdirective


