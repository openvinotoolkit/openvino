# ONNX format support in the OpenVINO™ {#openvino_docs_IE_DG_ONNX_Support}

Starting from the 2020.4 release, OpenVINO™ supports reading native ONNX models.
`Core::ReadNetwork()` method provides a uniform way to read models from IR or ONNX format, it is a recommended approach to reading models.
Example:

```cpp
InferenceEngine::Core core;
auto network = core.ReadNetwork("model.onnx");
```

**Reshape feature:**

OpenVINO™ doesn't provide a mechanism to specify pre-processing (like mean values subtraction, reverse input channels) for the ONNX format.
If an ONNX model contains dynamic shapes for input, please use the `CNNNetwork::reshape` method for shape specialization.

**Weights saved in external files:**

OpenVINO™ supports ONNX models that store weights in external files. It is especially useful for models larger than 2GB because of protobuf limitations.
To read such models, use the `ReadNetwork` overload which takes `modelPath` as input parameter (both `std::string` and `std::wstring`).
Note that the `binPath` argument of `ReadNetwork` should be empty in this case, because paths to external weights are saved directly in an ONNX model.
Otherwise, a runtime exception is thrown.
Reading models with external weights is not supported by the `ReadNetwork(const std::string& model, const Blob::CPtr& weights)` overload.

Paths to external weight files are saved in an ONNX model; these paths are relative to the model's directory path.
It means that if a model is located at:
`home/user/workspace/models/model.onnx`
and a file that contains external weights:
`home/user/workspace/models/data/weights.bin`
the path saved in model should be:
`data/weights.bin`.

**NOTE**

* A single model can use many external weights files.
* Data of many tensors can be stored in a single external weights file (it is processed using offset and length values, which can be also saved in a model).

The described mechanism is the only possibility to read weights from external files. The following input parameters of the `ReadNetwork` function overloads are NOT supported for ONNX models and should be passed as empty:
* `const std::wstring& binPath`
* `const std::string& binPath`
* `const Blob::CPtr& weights`

You can find more details about external data mechanism in [ONNX documentation](https://github.com/onnx/onnx/blob/master/docs/ExternalData.md).
To convert a model to use external data feature, you can use [ONNX helpers functions](https://github.com/onnx/onnx/blob/master/onnx/external_data_helper.py).

**Unsupported types of tensors:**

* `string`,
* `complex64`,
* `complex128`.
