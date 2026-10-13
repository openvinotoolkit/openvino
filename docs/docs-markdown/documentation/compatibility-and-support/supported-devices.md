---
sidebar_position: 1
sidebar_label: "Supported Devices"
description: "Check the list of devices used by OpenVINO to run inference of deep learning models."
---

# Supported Devices

The OpenVINO™ runtime enables you to use the following devices to run your
deep learning models:
[CPU](../../openvino-workflow/running-inference/inference-devices-and-modes/cpu-device/index.md),
[GPU](../../openvino-workflow/running-inference/inference-devices-and-modes/gpu-device/index.md),
[NPU](../../openvino-workflow/running-inference/inference-devices-and-modes/npu-device/index.md).

For their usage guides, see [Devices and Modes](../../openvino-workflow/running-inference/inference-devices-and-modes/index.md).  
For a detailed list of devices, see [System Requirements](../../about-openvino/release-notes-openvino/system-requirements.md).

Beside running inference with a specific device,
OpenVINO offers the option of running automated inference with the following inference modes:

[Automatic Device Selection](../../openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection/index.md):  
automatically selects the best device available for the given task. It offers many
additional options and optimizations, including inference on multiple devices at the
same time.

[Heterogeneous Inference](../../openvino-workflow/running-inference/inference-devices-and-modes/hetero-execution.md):  
enables splitting inference among several devices automatically, for example, if one device
doesn't support certain operations.

[Automatic Batching](../../openvino-workflow/running-inference/inference-devices-and-modes/automatic-batching.md):  
automatically groups inference requests to improve device utilization.

## Feature Support and API Coverage

| Supported Feature | CPU | GPU | NPU |
| --- | --- | --- | --- |
| [Automatic Device Selection](../../openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection/index.md) | Yes | Yes | Partial |
| [Heterogeneous execution](../../openvino-workflow/running-inference/inference-devices-and-modes/hetero-execution.md) | Yes | Yes | No |
| [Automatic batching](../../openvino-workflow/running-inference/inference-devices-and-modes/automatic-batching.md) | No | Yes | No |
| [Multi-stream execution](../../openvino-workflow/running-inference/optimize-inference/optimizing-throughput/index.md) | Yes | Yes | No |
| [Model caching](../../openvino-workflow/running-inference/optimize-inference/optimizing-latency/model-caching-overview.md) | Yes | Partial | Yes |
| [Dynamic shapes](../../openvino-workflow/running-inference/model-input-output/dynamic-shapes/index.md) | Yes | Partial | No |
| [Preprocessing acceleration](../../openvino-workflow/running-inference/optimize-inference/optimize-preprocessing/index.md) | Yes | Yes | No |
| [Stateful models](../../openvino-workflow/running-inference/inference-request/stateful-models/index.md) | Yes | Yes | Yes |
| [Extensibility](../openvino-extensibility/index.md) | Yes | Yes | No |

<table>
  <thead>
    <tr>
      <th><strong>API Coverage:</strong></th>
      <th>plugin</th>
      <th>infer_request</th>
      <th>compiled_model</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>CPU</td>
      <td>98.31 %</td>
      <td>100.0 %</td>
      <td>90.7 %</td>
    </tr>
    <tr>
      <td>CPU_ARM</td>
      <td>80.0 %</td>
      <td>100.0 %</td>
      <td>89.74 %</td>
    </tr>
    <tr>
      <td>GPU</td>
      <td>91.53 %</td>
      <td>100.0 %</td>
      <td>100.0 %</td>
    </tr>
    <tr>
      <td>dGPU</td>
      <td>89.83 %</td>
      <td>100.0 %</td>
      <td>100.0 %</td>
    </tr>
    <tr>
      <td>NPU</td>
      <td>18.64 %</td>
      <td>0.0 %</td>
      <td>9.3 %</td>
    </tr>
    <tr>
      <td>AUTO</td>
      <td>93.88 %</td>
      <td>100.0 %</td>
      <td>100.0 %</td>
    </tr>
    <tr>
      <td>BATCH</td>
      <td>86.05 %</td>
      <td>100.0 %</td>
      <td>86.05 %</td>
    </tr>
    <tr>
      <td>HETERO</td>
      <td>61.22 %</td>
      <td>99.24 %</td>
      <td>86.05 %</td>
    </tr>
    <tr>
      <td></td>
      <td colspan="3">Percentage of API supported by the device,<br/>as of OpenVINO 2024.5, 20 Nov. 2024.</td>
    </tr>
  </tbody>
</table>

For setting up a relevant configuration, refer to the
[Integrate with Customer Application](../../openvino-workflow/running-inference/index.md)
topic (step 3 "Configure input and output").

<details>
<summary>Device support across OpenVINO 2024.6 distributions</summary>

| Device | Archives | PyPI | APT/YUM/ZYPPER | Conda | Homebrew | vcpkg | Conan | npm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CPU | V | V | V | V | V | V | V | V |
| GPU | V | V | V | V | V | V | V | V |
| NPU | V\* | V\* | V \* | n/a | n/a | n/a | n/a | V\* |

\* **Of the Linux systems, versions 22.04 and 24.04 include drivers for NPU.**  
**For Windows, CPU inference on ARM64 is not supported.**

</details>
