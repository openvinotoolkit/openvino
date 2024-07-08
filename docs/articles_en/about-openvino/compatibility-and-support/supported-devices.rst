Supported Inference Devices
============================

.. meta::
   :description: Check the list of devices used by OpenVINO to run inference
                 of deep learning models.


The OpenVINO™ runtime enables you to use a selection of devices to run your
deep learning models:
:doc:`CPU <../../openvino-workflow/running-inference/inference-devices-and-modes/cpu-device>`,
:doc:`GPU <../../openvino-workflow/running-inference/inference-devices-and-modes/gpu-device>`,
:doc:`NPU <../../openvino-workflow/running-inference/inference-devices-and-modes/npu-device>`.

| For their usage guides, see :doc:`Devices and Modes <../../openvino-workflow/running-inference/inference-devices-and-modes>`.
| For a detailed list of devices, see :doc:`System Requirements <../release-notes-openvino/system-requirements>`.

Beside running inference with a specific device,
OpenVINO offers the option of running automated inference with the following inference modes:

* :doc:`Automatic Device Selection <../../openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection>` - automatically selects the best device
  available for the given task. It offers many additional options and optimizations, including inference on
  multiple devices at the same time.
* :doc:`Heterogeneous Inference <../../openvino-workflow/running-inference/inference-devices-and-modes/hetero-execution>` - enables splitting inference among several devices
  automatically, for example, if one device doesn't support certain operations.
* :doc:`Multi-device Inference <../../openvino-workflow/running-inference/inference-devices-and-modes/multi-device>` - executes inference on multiple devices.
  Currently, this mode is considered a legacy solution. Using Automatic Device Selection is advised.
* :doc:`Automatic Batching <../../openvino-workflow/running-inference/inference-devices-and-modes/automatic-batching>` - automatically groups inference requests to improve
  device utilization.



Feature Support and API Coverage
#################################

=============================================================================================================================== ======= ========== ===========
 Supported Feature                                                                                                               CPU     GPU        NPU
=============================================================================================================================== ======= ========== ===========
 :doc:`Heterogeneous execution <../../openvino-workflow/running-inference/inference-devices-and-modes/hetero-execution>`         Yes     Yes        No
 :doc:`Multi-device execution <../../openvino-workflow/running-inference/inference-devices-and-modes/multi-device>`              Yes     Yes        Partial
 :doc:`Automatic batching <../../openvino-workflow/running-inference/inference-devices-and-modes/automatic-batching>`            No      Yes        No
 :doc:`Multi-stream execution <../../openvino-workflow/running-inference/optimize-inference/optimizing-throughput>`              Yes     Yes        No
 :doc:`Models caching <../../openvino-workflow/running-inference/optimize-inference/optimizing-latency/model-caching-overview>`  Yes     Partial    Yes
 :doc:`Dynamic shapes <../../openvino-workflow/running-inference/dynamic-shapes>`                                                Yes     Partial    No
 :doc:`Import/Export <../../documentation/openvino-ecosystem>`                                                                   Yes     Yes        Yes
 :doc:`Preprocessing acceleration <../../openvino-workflow/running-inference/optimize-inference/optimize-preprocessing>`         Yes     Yes        No
 :doc:`Stateful models <../../openvino-workflow/running-inference/stateful-models>`                                              Yes     Yes        Yes
 :doc:`Extensibility <../../documentation/openvino-extensibility>`                                                               Yes     Yes        No
=============================================================================================================================== ======= ========== ===========


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

For setting up a relevant configuration, refer to the
:doc:`Integrate with Customer Application <../../openvino-workflow/running-inference/integrate-openvino-with-your-application>`
topic (step 3 "Configure input and output").



.. note::

   With OpenVINO 2024.0 release, support for GNA has been discontinued. To keep using it
   in your solutions, revert to the 2023.3 (LTS) version.

   With OpenVINO™ 2023.0 release, support has been cancelled for:
   - Intel® Neural Compute Stick 2 powered by the Intel® Movidius™ Myriad™ X
   - Intel® Vision Accelerator Design with Intel® Movidius™

   To keep using the MYRIAD and HDDL plugins with your hardware,
   revert to the OpenVINO 2022.3 (LTS) version.
