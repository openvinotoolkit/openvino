Supported Devices
===============================================================================================

.. meta::
   :description: Check the list of devices used by OpenVINO to run inference
                 of deep learning models.


The OpenVINO™ runtime enables you to use the following devices to run your
deep learning models:
:doc:`CPU <../../openvino-workflow/running-inference/inference-devices-and-modes/cpu-device>`,
:doc:`GPU <../../openvino-workflow/running-inference/inference-devices-and-modes/gpu-device>`,
:doc:`NPU <../../openvino-workflow/running-inference/inference-devices-and-modes/npu-device>`.

| For their usage guides, see :doc:`Devices and Modes <../../openvino-workflow/running-inference/inference-devices-and-modes>`.
| For a detailed list of devices, see :doc:`System Requirements <../release-notes-openvino/system-requirements>`.

Beside running inference with a specific device,
OpenVINO offers the option of running automated inference with the following inference modes:

| :doc:`Automatic Device Selection <../../openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection>`:
|     automatically selects the best device available for the given task. It offers many
      additional options and optimizations, including inference on multiple devices at the
      same time.
| :doc:`Heterogeneous Inference <../../openvino-workflow/running-inference/inference-devices-and-modes/hetero-execution>`:
|     enables splitting inference among several devices automatically, for example, if one device
      doesn't support certain operations.

| :doc:`Automatic Batching <../../openvino-workflow/running-inference/inference-devices-and-modes/automatic-batching>`:
|     automatically groups inference requests to improve device utilization.

| :doc:`(LEGACY) Multi-device Inference <./../../documentation/legacy-features/multi-device>`:
|     executes inference on multiple devices. Currently, this mode is considered a legacy
      solution. Using Automatic Device Selection instead is advised.


Feature Support and API Coverage
#################################

======================================================================================================================================== ======= ========== ===========
 Supported Feature                                                                                                                        CPU     GPU        NPU
======================================================================================================================================== ======= ========== ===========
 :doc:`Automatic Device Selection <../../openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection>`          Yes     Yes        Partial
 :doc:`Heterogeneous execution <../../openvino-workflow/running-inference/inference-devices-and-modes/hetero-execution>`                  Yes     Yes        No
 :doc:`Automatic batching <../../openvino-workflow/running-inference/inference-devices-and-modes/automatic-batching>`                     No      Yes        No
 :doc:`Multi-stream execution <../../openvino-workflow/running-inference/optimize-inference/optimizing-throughput>`                       Yes     Yes        No
 :doc:`Model caching <../../openvino-workflow/running-inference/optimize-inference/optimizing-latency/model-caching-overview>`            Yes     Partial    Yes
 :doc:`Dynamic shapes <../../openvino-workflow/running-inference/dynamic-shapes>`                                                         Yes     Partial    No
 :doc:`Import/Export <../../documentation/openvino-ecosystem>`                                                                            Yes     Yes        Yes
 :doc:`Preprocessing acceleration <../../openvino-workflow/running-inference/optimize-inference/optimize-preprocessing>`                  Yes     Yes        No
 :doc:`Stateful models <../../openvino-workflow/running-inference/stateful-models>`                                                       Yes     Yes        Yes
 :doc:`Extensibility <../../documentation/openvino-extensibility>`                                                                        Yes     Yes        No
 :doc:`(LEGACY) Multi-device execution <./../../documentation/legacy-features/multi-device>`                                              Yes     Yes        Partial
======================================================================================================================================== ======= ========== ===========


+-------------------------+-----------+------------------+-------------------+
| **API Coverage:**       | plugin    | infer_request    | compiled_model    |
+=========================+===========+==================+===================+
| CPU                     | 98.31 %   | 100.0 %          | 90.7 %            |
+-------------------------+-----------+------------------+-------------------+
| CPU_ARM                 | 80.0 %    | 100.0 %          | 89.74 %           |
+-------------------------+-----------+------------------+-------------------+
| GPU                     | 91.53 %   | 100.0 %          | 100.0 %           |
+-------------------------+-----------+------------------+-------------------+
| dGPU                    | 89.83 %   | 100.0 %          | 100.0 %           |
+-------------------------+-----------+------------------+-------------------+
| NPU                     | 18.64 %   | 0.0 %            | 9.3 %             |
+-------------------------+-----------+------------------+-------------------+
| AUTO                    | 93.88 %   | 100.0 %          | 100.0 %           |
+-------------------------+-----------+------------------+-------------------+
| BATCH                   | 86.05 %   | 100.0 %          | 86.05 %           |
+-------------------------+-----------+------------------+-------------------+
| HETERO                  | 61.22 %   | 99.24 %          | 86.05 %           |
+-------------------------+-----------+------------------+-------------------+
|                         || Percentage of API supported by the device,      |
|                         || as of OpenVINO 2023.3, 08 Jan, 2024.            |
+-------------------------+-----------+------------------+-------------------+

For setting up a relevant configuration, refer to the
:doc:`Integrate with Customer Application <../../openvino-workflow/running-inference/integrate-openvino-with-your-application>`
topic (step 3 "Configure input and output").



.. note::

   With the OpenVINO 2024.0 release, support for GNA has been discontinued. To keep using it
   in your solutions, revert to the 2023.3 (LTS) version.

   With the OpenVINO™ 2023.0 release, support has been cancelled for:
   - Intel® Neural Compute Stick 2 powered by the Intel® Movidius™ Myriad™ X
   - Intel® Vision Accelerator Design with Intel® Movidius™

   To keep using the MYRIAD and HDDL plugins with your hardware,
   revert to the OpenVINO 2022.3 (LTS) version.
