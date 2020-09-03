# ONNX format support in the OpenVINO™ {#openvino_docs_IE_DG_ONNX_Support}

Starting from the 2020.4 release, OpenVINO™ supports reading native ONNX models.
`Core::ReadNetwork()` method provides a uniform way to read models from IR or ONNX format, it is a recommended approach to reading models.
Example:

```cpp
InferenceEngine::Core core;
auto network = core.ReadNetwork("model.onnx");
```

OpenVINO™ doesn't provide a mechanism to specify pre-processing (like mean values subtraction, reverse input channels) for the ONNX format.
If an ONNX model contains dynamic shapes for input, please use the `CNNNetwork::reshape` method for shape specialization.

Unsupported types of tensors:

* `string`,
* `complex64`,
* `complex128`.
