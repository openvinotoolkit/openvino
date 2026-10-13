---
sidebar_position: 5
sidebar_label: "Overview of OpenVINO Plugin Library"
description: "Develop and implement independent inference solutions for different devices with the components of plugin architecture of OpenVINO."
---

# Overview of OpenVINO Plugin Library


The plugin architecture of OpenVINO allows to develop and plug independent inference
solutions dedicated to different devices. Physically, a plugin is represented as a dynamic library
exporting the single `create_plugin_engine` function that allows to create a new plugin instance.

## OpenVINO Plugin Library

OpenVINO plugin dynamic library consists of several main components:

1. [Plugin class](plugin.md):
   
   - Provides information about devices of a specific type.
   - Can create an  [compiled model](compiled-model.md) instance which represents a Neural Network backend specific graph structure for a particular device in opposite to the ov::Model which is backend-independent.
   - Can import an already compiled graph structure from an input stream to a [compiled model](compiled-model.md) object.
2. [Compiled Model class](compiled-model.md):
   
   - Is an execution configuration compiled for a particular device and takes into account its capabilities.
   - Holds a reference to a particular device and a task executor for this device.
   - Can create several instances of  [Inference Request](synch-inference-request.md).
   - Can export an internal backend specific graph structure to an output stream.
3. [Inference Request class](synch-inference-request.md):
   
   - Runs an inference pipeline serially.
   - Can extract performance counters for an inference pipeline execution profiling.
4. [Asynchronous Inference Request class](asynch-inference-request.md):
   
   - Wraps the  [Inference Request](synch-inference-request.md) class and runs pipeline stages in parallel on several task executors based on a device-specific pipeline structure.
5. [Plugin specific properties](plugin-properties.md):
   
   - Provides the plugin specific properties.
6. [Remote Context](remote-context.md):
   
   - Provides the device specific remote context. Context allows to create remote tensors.
7. [Remote Tensor](remote-tensor.md)
   
   - Provides the device specific remote tensor API and implementation.

:::note
This documentation is written based on the `Template` plugin, which demonstrates plugin development details. Find the complete code of the `Template`, which is fully compilable and up-to-date, at `<openvino source dir>/src/plugins/template`.
:::

## Detailed Guides

- [Build](build-plugin-using-cmake.md) a plugin library using CMake
- Plugin and its components [testing](plugin-testing.md)
- [Quantized networks](advanced-guides/quantized-models.md)
- [Low precision transformations](advanced-guides/low-precision-transformations/index.md) guide
- [Writing OpenVINO™ transformations](../transformation-api/index.md) guide
- [Integration with AUTO Plugin](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/auto/docs/integration.md)

## API References

- [OpenVINO Plugin API](https://docs.openvino.ai/2026/api/c_cpp_api/group__ov__dev__api.html)
- [OpenVINO Transformation API](https://docs.openvino.ai/2026/api/c_cpp_api/group__ie__transformation__api.html)
