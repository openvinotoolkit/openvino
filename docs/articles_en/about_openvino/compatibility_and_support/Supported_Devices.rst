.. {#openvino_supported_devices}


Inference Device Support
========================

.. meta::
   :description: Check the list of devices used by OpenVINO to run inference
                 of deep learning models.


The OpenVINO™ runtime enables you to use a selection of devices to run your
deep learning models:
:doc:`CPU <openvino_docs_OV_UG_supported_plugins_CPU>`,
:doc:`GPU <openvino_docs_OV_UG_supported_plugins_GPU>`,
:doc:`NPU <openvino_docs_OV_UG_supported_plugins_NPU>`.

| For their usage guides, see :doc:`Devices and Modes <openvino_docs_Runtime_Inference_Modes_Overview>`.
| For a detailed list of devices, see :doc:`System Requirements <system_requirements>`.

Beside running inference with a specific device,
OpenVINO offers the option of running automated inference with the following inference modes:

* :doc:`Automatic Device Selection <openvino_docs_OV_UG_supported_plugins_AUTO>` - automatically selects the best device
  available for the given task. It offers many additional options and optimizations, including inference on
  multiple devices at the same time.
* :doc:`Heterogeneous Inference <openvino_docs_OV_UG_Hetero_execution>` - enables splitting inference among several devices
  automatically, for example, if one device doesn't support certain operations.
* :doc:`Multi-device Inference <openvino_docs_OV_UG_Running_on_multiple_devices>` - executes inference on multiple devices.
  Currently, this mode is considered a legacy solution. Using Automatic Device Selection is advised.
* :doc:`Automatic Batching <openvino_docs_OV_UG_Automatic_Batching>` - automatically groups inference requests to improve
  device utilization.



Feature Support and API Coverage
#################################

================================================================================== ======= ========== ===========
 Supported Feature                                                                  CPU     GPU        NPU
================================================================================== ======= ========== ===========
 :doc:`Heterogeneous execution <openvino_docs_OV_UG_Hetero_execution>`              Yes     Yes        No
 :doc:`Multi-device execution <openvino_docs_OV_UG_Running_on_multiple_devices>`    Yes     Yes        Partial
 :doc:`Automatic batching <openvino_docs_OV_UG_Automatic_Batching>`                 No      Yes        No
 :doc:`Multi-stream execution <openvino_docs_deployment_optimization_guide_tput>`   Yes     Yes        No
 :doc:`Models caching <openvino_docs_OV_UG_Model_caching_overview>`                 Yes     Partial    Yes
 :doc:`Dynamic shapes <openvino_docs_OV_UG_DynamicShapes>`                          Yes     Partial    No
 :doc:`Import/Export <openvino_ecosystem>`                                          Yes     No         Yes
 :doc:`Preprocessing acceleration <openvino_docs_OV_UG_Preprocessing_Overview>`     Yes     Yes        No
 :doc:`Stateful models <openvino_docs_OV_UG_stateful_models_intro>`                 Yes     No         Yes
 :doc:`Extensibility <openvino_docs_Extensibility_UG_Intro>`                        Yes     Yes        No
================================================================================== ======= ========== ===========


+-------------------------+-----------+------------------+-------------------+
| **API Coverage:**       | plugin    | infer_request    | compiled_model    |
+=========================+===========+==================+===================+
| CPU                     | 80.0 %    | 100.0 %          | 89.74 %           |
+-------------------------+-----------+------------------+-------------------+
| CPU_ARM                 | 80.0 %    | 100.0 %          | 89.74 %           |
+-------------------------+-----------+------------------+-------------------+
| GPU                     | 84.0 %    | 100.0 %          | 100.0 %           |
+-------------------------+-----------+------------------+-------------------+
| dGPU                    | 82.0 %    | 100.0 %          | 100.0 %           |
+-------------------------+-----------+------------------+-------------------+
| NPU                     | 16.0 %    | 0.0 %            | 10.26 %           |
+-------------------------+-----------+------------------+-------------------+
| AUTO                    | 40.0 %    | 100.0 %          | 97.44 %           |
+-------------------------+-----------+------------------+-------------------+
| BATCH                   | 26.0 %    | 100.0 %          | 58.97 %           |
+-------------------------+-----------+------------------+-------------------+
| MULTI                   | 30.0 %    | 100.0 %          | 58.97 %           |
+-------------------------+-----------+------------------+-------------------+
| HETERO                  | 30.0 %    | 99.23 %          | 58.97 %           |
+-------------------------+-----------+------------------+-------------------+
|                         || Percentage of API supported by the device,      |
|                         || as of OpenVINO 2023.3, 08 Jan, 2024.            |
+-------------------------+-----------+------------------+-------------------+


Devices similar to the ones used for benchmarking can be accessed using
`Intel® DevCloud for the Edge <https://devcloud.intel.com/edge/>`__,
a remote development environment with access to Intel® hardware and the latest versions
of the Intel® Distribution of OpenVINO™ Toolkit.
`Learn more <https://devcloud.intel.com/edge/get_started/devcloud/>`__ or
`Register here <https://inteliot.force.com/DevcloudForEdge/s/>`__.

For setting up a relevant configuration, refer to the
:doc:`Integrate with Customer Application <openvino_docs_OV_UG_Integrate_OV_with_your_application>`
topic (step 3 "Configure input and output").



.. note::

   With OpenVINO 2024.0 release, support for GNA has been discontinued. To keep using it
   in your solutions, revert to the 2023.3 (LTS) version.

   With OpenVINO™ 2023.0 release, support has been cancelled for:
   - Intel® Neural Compute Stick 2 powered by the Intel® Movidius™ Myriad™ X
   - Intel® Vision Accelerator Design with Intel® Movidius™

   To keep using the MYRIAD and HDDL plugins with your hardware,
   revert to the OpenVINO 2022.3 (LTS) version.
