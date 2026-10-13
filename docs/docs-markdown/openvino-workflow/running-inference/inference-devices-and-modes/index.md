---
sidebar_position: 4
sidebar_label: "Inference Devices and Modes"
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Inference Devices and Modes


The OpenVINO™ Runtime offers several inference modes to optimize hardware usage.
You can run inference on a single device or use automated modes that manage multiple devices:

**single-device inference**  
This mode runs all inference on one selected device. The OpenVINO Runtime includes
built-in plugins that support the following devices:  
[CPU](cpu-device/index.md)  
[GPU](gpu-device/index.md)  
[NPU](npu-device/index.md)

**automated inference modes**  
These modes automate device selection and workload distribution, potentially increasing
performance and portability:  
[Automatic Device Selection (AUTO)](auto-device-selection/index.md)  
[Heterogeneous Execution (HETERO)](hetero-execution.md)  across different device types  
[Automatic Batching Execution (Auto-batching)](automatic-batching.md): automatically groups inference requests to improve throughput

Learn how to configure devices in the [Query device properties](query-device-properties.md) article.

## Enumerating Available Devices

The OpenVINO Runtime API provides methods to list available devices and their details.
When there are multiple instances of a device, they get specific names like GPU.0 for iGPU.
Here is an example of the output with device names, including two GPUs:

```bash
./hello_query_device
Available devices:
    Device: CPU
...
    Device: GPU.0
...
    Device: GPU.1
```

See the [Hello Query Device Sample](../../../get-started/learn-openvino/openvino-samples/hello-query-device.md)
for more details.

Below is an example showing how to list available devices and use them with multi-device mode:

<Tabs>
  <TabItem value="cpp" label="C++">
  ```cpp
  ov::Core core;
  std::shared_ptr<ov::Model> model = core.read_model("sample.xml");
  std::vector<std::string> availableDevices = core.get_available_devices();
  std::string all_devices;
  for (auto && device : availableDevices) {
      all_devices += device;
      all_devices += ((device == availableDevices[availableDevices.size()-1]) ? "" : ",");
  }
  ov::CompiledModel compileModel = core.compile_model(model, "MULTI",
      ov::device::priorities(all_devices));
  ```
  </TabItem>
</Tabs>

If you have two GPU devices, you can specify them explicitly as “MULTI:GPU.1,GPU.0”.
Here is how to list and use all available GPU devices:

<Tabs>
  <TabItem value="cpp" label="C++">
  ```cpp
  ov::Core core;
  std::vector<std::string> GPUDevices = core.get_property("GPU", ov::available_devices);
  std::string all_devices;
  for (size_t i = 0; i < GPUDevices.size(); ++i) {
      all_devices += std::string("GPU.")
                              + GPUDevices[i]
                              + std::string(i < (GPUDevices.size() -1) ? "," : "");
  }
  ov::CompiledModel compileModel = core.compile_model("sample.xml", "MULTI",
      ov::device::priorities(all_devices));
  ```
  </TabItem>
</Tabs>

## Additional Resources

- [OpenVINO™ Runtime API Tutorial](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/openvino-api)
- [AUTO Device Tutorial](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/auto-device)
- [GPU Device Tutorial](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/gpu-device)
- [NPU Device Tutorial](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/hello-npu)
