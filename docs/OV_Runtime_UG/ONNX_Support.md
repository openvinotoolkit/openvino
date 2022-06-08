# ONNX Format Support {#ONNX_Format_Support}


Since the 2020.4 release, OpenVINO™ has supported native usage of ONNX models. The `core.read_model()` method, which is the recommended approach to reading models, provides a uniform way to work with OpenVINO IR and ONNX formats alike. Example:

@sphinxdirective
.. tab:: C++

   .. code-block:: cpp
   
      ov::Core core;
      std::shared_ptr<ov::Model> model = core.read_model("model.xml")
 
.. tab:: Python

   .. code-block:: python

      import openvino.runtime as ov
      core = ov.Core()
      model = core.read_model("model.xml")
@endsphinxdirective

While ONNX models are directly supported by OpenVINO™, it can be useful to convert them to IR format to take advantage of advanced OpenVINO optimization tools and features. For information on how to convert an ONNX model to the OpenVINO IR format, see the [Converting an ONNX Model](https://github.com/openvinotoolkit/openvino/pull/MO_DG/prepare_model/convert_model/Convert_Model_From_ONNX.md) page.
### Reshape Feature
OpenVINO™ does not provide a mechanism to specify pre-processing for the ONNX format, like mean value subtraction or reverse input channels. If an ONNX model contains dynamic shapes for input, please see the [Changing input shapes](ShapeInference.md) documentation.

### Weights Saved in External Files
OpenVINO™ supports ONNX models that store weights in external files. It is especially useful for models larger than 2GB because of protobuf limitations. To read such models:

@sphinxdirective
.. tab:: C++

   * Use the `read_model` overload that takes `modelPath` as the input parameter (both `std::string` and `std::wstring`).
   * The `binPath` argument of `read_model` should be empty. Otherwise, a runtime exception is thrown because paths to external weights are saved directly in the ONNX model.
   * Reading models with external weights is **NOT** supported by the `read_model()` overload.
  
.. tab:: Python

   * Use the `model` parameter in the `openvino.runtime.Core.read_model(model : "path_to_onnx_file")` method.
   * The `weights` parameter, for the path to the binary weight file, should be empty. Otherwise, a runtime exception is thrown because paths to external weights are saved directly in the ONNX model.
   * Reading models with external weights is **NOT** supported by the `read_model(weights: "path_to_bin_file")` parameter.
   
@endsphinxdirective

Paths to external weight files are saved in an ONNX model. They are relative to the model's directory path, which means that for a model located at `workspace/models/model.onnx` and a weights file at `workspace/models/data/weights.bin`, the path saved in the model would be: `data/weights.bin`.

Note that a single model can use many external weights files.
What is more, data of many tensors can be stored in a single external weights file, processed using offset and length values, which can also be saved in a model.

The following input parameters are NOT supported for ONNX models and should be passed as empty (none) or not at all:

* for `ReadNetwork` (C++):
   * `const std::wstring& binPath`
   * `const std::string& binPath`
   * `const Tensor& weights`
* for [openvino.runtime.Core.read_model](https://docs.openvino.ai/latest/api/ie_python_api/_autosummary/openvino.runtime.Core.html#openvino.runtime.Core.read_model)
   * `weights`


You can find more details about the external data mechanism in [ONNX documentation](https://github.com/onnx/onnx/blob/master/docs/ExternalData.md).
To convert a model to use the external data feature, you can use [ONNX helper functions](https://github.com/onnx/onnx/blob/master/onnx/external_data_helper.py).

Unsupported types of tensors:
* string
* complex64
* complex128
