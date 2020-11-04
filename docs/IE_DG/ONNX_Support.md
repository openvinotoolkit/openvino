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

OpenVINO™ supports ONNX models which use weights saved in external files. It is especially useful for models larger than 2GB because of protobuf limitations.
If you want to read such model you should use ReadNetwork overload which takes `modelPath` as input parameter (both `std::string` and `std::wstring`).
The feature is not supported by `ReadNetwork(const std::string& model, const Blob::CPtr& weights)` overload.

The paths to external weights files are saved directly in the model file as relative to the model location.
It means that if there is a model:
`home/user/workspace/models/model.onnx`
and a file which contains external weights:
`home/user/workspace/weights.data`
the path saved in model should be:
`../../weights.data`.

**NOTE**

* A single model can use many external weights files.
* In a single external weights file can be saved data to many tensors (it is processed using offset and length values which can be also saved in a model).

The described mechanism is the only possibility to read weights from external files. The following input parameters of `ReadNetwork` function overloads are NOT supported for ONNX models and should be passed as empty (or `nullptr`):
* `const std::wstring& binPath`
* `const std::string& binPath`
* `const Blob::CPtr& weights`

More details about external data mechanism you can find in [ONNX documentation](https://github.com/onnx/onnx/blob/master/docs/ExternalData.md).
In order to convert a model to use external data feature you can use [ONNX helpers functions](https://github.com/onnx/onnx/blob/master/onnx/external_data_helper.py).

**Unsupported types of tensors:**

* `string`,
* `complex64`,
* `complex128`.
