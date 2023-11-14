# Supported Devices {#openvino_docs_OV_UG_supported_plugins_Supported_Devices}


@sphinxdirective

.. meta::
   :description: Check the list of officially supported models in Intel® 
                 Distribution of OpenVINO™ toolkit.


The OpenVINO runtime can infer various models of different input and output formats. Here, you can find configurations 
supported by OpenVINO devices, which are CPU, GPU, NPU, and GNA (Gaussian Neural Accelerator coprocessor).
Currently, processors of the 11th generation and later (up to the 13th generation at the moment) provide a further performance boost, especially with INT8 models.

.. note::

   With OpenVINO™ 2023.0 release, support has been cancelled for:
   - Intel® Neural Compute Stick 2 powered by the Intel® Movidius™ Myriad™ X
   - Intel® Vision Accelerator Design with Intel® Movidius™
   
   To keep using the MYRIAD and HDDL plugins with your hardware, revert to the OpenVINO 2022.3 LTS release.
   


+---------------------------------------------------------------------+------------------------------------------------------------------------------------------------------+
| OpenVINO Device                                                     | Supported Hardware                                                                                   |
+=====================================================================+======================================================================================================+
|| :doc:`CPU <openvino_docs_OV_UG_supported_plugins_CPU>`             | Intel® Xeon® with Intel® Advanced Vector Extensions 2 (Intel® AVX2), Intel® Advanced Vector          |
||   (x86)                                                            | Extensions 512 (Intel® AVX-512), Intel® Advanced Matrix Extensions (Intel® AMX),                     | 
||                                                                    | Intel® Core™ Processors with Intel® AVX2,                                                            |
||                                                                    | Intel® Atom® Processors with Intel® Streaming SIMD Extensions (Intel® SSE)                           |
||                                                                    |                                                                                                      |
||   (Arm®)                                                           | Raspberry Pi™ 4 Model B, Apple® Mac mini with Apple silicon                                          |
||                                                                    |                                                                                                      |
+---------------------------------------------------------------------+------------------------------------------------------------------------------------------------------+
|| :doc:`GPU <openvino_docs_OV_UG_supported_plugins_GPU>`             | Intel® Processor Graphics including Intel® HD Graphics and Intel® Iris® Graphics,                    |
||                                                                    | Intel® Arc™ A-Series Graphics, Intel® Data Center GPU Flex Series, Intel® Data Center GPU Max Series |                                 
+---------------------------------------------------------------------+------------------------------------------------------------------------------------------------------+
|| :doc:`GNA <openvino_docs_OV_UG_supported_plugins_GNA>`             | Intel® Speech Enabling Developer Kit, Amazon Alexa* Premium Far-Field Developer Kit, Intel®          |
|| (available in the Intel® Distribution of OpenVINO™ toolkit)        | Pentium® Silver J5005 Processor, Intel® Pentium® Silver N5000 Processor, Intel®                      |
||                                                                    | Celeron® J4005 Processor, Intel® Celeron® J4105 Processor, Intel® Celeron®                           |
||                                                                    | Processor N4100, Intel® Celeron® Processor N4000, Intel® Core™ i3-8121U Processor,                   |
||                                                                    | Intel® Core™ i7-1065G7 Processor, Intel® Core™ i7-1060G7 Processor, Intel®                           |
||                                                                    | Core™ i5-1035G4 Processor, Intel® Core™ i5-1035G7 Processor, Intel® Core™                            |
||                                                                    | i5-1035G1 Processor, Intel® Core™ i5-1030G7 Processor, Intel® Core™ i5-1030G4 Processor,             |
||                                                                    | Intel® Core™ i3-1005G1 Processor, Intel® Core™ i3-1000G1 Processor,                                  |
||                                                                    | Intel® Core™ i3-1000G4 Processor                                                                     |
+---------------------------------------------------------------------+------------------------------------------------------------------------------------------------------+
|| NPU                                                                |                                                                                                      |
||                                                                    |                                                                                                      |
||                                                                    |                                                                                                      |
||                                                                    |                                                                                                      |
||                                                                    |                                                                                                      |
||                                                                    |                                                                                                      |
||                                                                    |                                                                                                      |
||                                                                    |                                                                                                      |
+---------------------------------------------------------------------+------------------------------------------------------------------------------------------------------+

Beside inference using a specific device, OpenVINO offers three inference modes for automated inference management. These are:

* :doc:`Automatic Device Selection <openvino_docs_OV_UG_supported_plugins_AUTO>` - automatically selects the best device 
  available for the given task. It offers many additional options and optimizations, including inference on 
  multiple devices at the same time.
* :doc:`Multi-device Inference <openvino_docs_OV_UG_Running_on_multiple_devices>` - executes inference on multiple devices. 
  Currently, this mode is considered a legacy solution. Using Automatic Device Selection is advised.
* :doc:`Heterogeneous Inference <openvino_docs_OV_UG_Hetero_execution>` - enables splitting inference among several devices 
  automatically, for example, if one device doesn’t support certain operations.


Devices similar to the ones we have used for benchmarking can be accessed using `Intel® DevCloud for the Edge <https://devcloud.intel.com/edge/>`__, 
a remote development environment with access to Intel® hardware and the latest versions of the Intel® Distribution 
of OpenVINO™ Toolkit. `Learn more <https://devcloud.intel.com/edge/get_started/devcloud/>`__ or `Register here <https://inteliot.force.com/DevcloudForEdge/s/>`__.


To learn more about each of the supported devices and modes, refer to the sections of:
* :doc:`Inference Device Support <openvino_docs_OV_UG_Working_with_devices>` 
* :doc:`Inference Modes <openvino_docs_Runtime_Inference_Modes_Overview>`



For setting relevant configuration, refer to the
:doc:`Integrate with Customer Application <openvino_docs_OV_UG_Integrate_OV_with_your_application>` 
topic (step 3 "Configure input and output").



@endsphinxdirective


