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
| For a detailed list of devices, see :doc:`System Requirements <../../about-openvino/release-notes-openvino/system-requirements>`.


Beside running inference with a specific device,
OpenVINO offers the option of running automated inference with the following inference modes:

| :doc:`Automatic Device Selection <../../openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection>`:
| automatically selects the best device available for the given task. It offers many
  additional options and optimizations, including inference on multiple devices at the
  same time.

| :doc:`Heterogeneous Inference <../../openvino-workflow/running-inference/inference-devices-and-modes/hetero-execution>`:
| enables splitting inference among several devices automatically, for example, if one device
  doesn't support certain operations.

| :doc:`Automatic Batching <../../openvino-workflow/running-inference/inference-devices-and-modes/automatic-batching>`:
| automatically groups inference requests to improve device utilization.

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
 :doc:`Dynamic shapes <../../openvino-workflow/running-inference/model-input-output/dynamic-shapes>`                                      Yes     Partial    No
 :doc:`Preprocessing acceleration <../../openvino-workflow/running-inference/optimize-inference/optimize-preprocessing>`                  Yes     Yes        No
 :doc:`Stateful models <../../openvino-workflow/running-inference/inference-request/stateful-models>`                                     Yes     Yes        Yes
 :doc:`Extensibility <../../documentation/openvino-extensibility>`                                                                        Yes     Yes        No
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
|                         || as of OpenVINO 2024.5, 20 Nov. 2024.            |
+-------------------------+-----------+------------------+-------------------+

For setting up a relevant configuration, refer to the
:doc:`Integrate with Customer Application <../../openvino-workflow/running-inference>`
topic (step 3 "Configure input and output").

.. dropdown:: Device support across OpenVINO 2024.6 distributions

   ===============  ==========  ======  ===============  ========  ============ ========== ========== ==========
   Device           Archives    PyPI    APT/YUM/ZYPPER    Conda     Homebrew     vcpkg      Conan       npm
   ===============  ==========  ======  ===============  ========  ============ ========== ========== ==========
   CPU              V           V       V                V         V            V          V          V
   GPU              V           V       V                V         V            V          V          V
   NPU              V\*         V\*     V\ *             n/a       n/a          n/a        n/a        V\*
   ===============  ==========  ======  ===============  ========  ============ ========== ========== ==========

   | \* **Of the Linux systems, versions 22.04 and 24.04 include drivers for NPU.**
   |  **For Windows, CPU inference on ARM64 is not supported.**

