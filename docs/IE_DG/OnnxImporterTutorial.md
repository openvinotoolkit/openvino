# ONNX* Importer API Tutorial {#openvino_docs_IE_DG_OnnxImporterTutorial}

> **NOTE**: This tutorial is deprecated. Since OpenVINO™ 2020.4 version, Inference Engine enables reading ONNX models via the Inference Engine Core API
> and there is no need to use directly the low-level ONNX* Importer API anymore. 
> To read ONNX\* models, it's recommended to use the `Core::ReadNetwork()` method that provide a uniform way to read models from IR or ONNX format.

This tutorial demonstrates how to use the ONNX\* Importer API.
This API makes it possible to create an nGraph `Function` object from an imported ONNX model.

All functions of the ONNX Importer API are in the [onnx.hpp][onnx_header] header file.

Two categories of API functions:
* Helper functions that check which ONNX ops are supported in a current version of the ONNX Importer
* Functions that read ONNX models from a stream or file and result in an nGraph function, which can be executed using the Inference Engine

## Check Which ONNX Ops Are Supported

To list all supported ONNX ops in a specific version and domain, use the `get_supported_operators` 
as shown in the example below:

@snippet openvino/docs/snippets/OnnxImporterTutorial0.cpp part0

The above code produces a list of all the supported operators for the `version` and `domain` you specified and outputs a list similar to this:
```cpp
Abs
Acos
...
Xor
```

To determine whether a specific ONNX operator in a particular version and domain is supported by the importer, use the `is_operator_supported` function as shown in the example below:

@snippet openvino/docs/snippets/OnnxImporterTutorial1.cpp part1

## Import ONNX Model

To import an ONNX model, use the `import_onnx_model` function.
The method has two overloads:
* <a href="#stream">`import_onnx_model` takes a stream as an input</a>, for example, file stream, memory stream
* <a href="#path">`import_onnx_model` takes a file path as an input</a>

Refer to the sections below for details.

> **NOTE**: The examples below use the ONNX ResNet50 model, which is available at the [ONNX Model Zoo][onnx_model_zoo]:
> ```bash
> $ wget https://s3.amazonaws.com/download.onnx/models/opset_8/resnet50.tar.gz
> $ tar -xzvf resnet50.tar.gz
> ```

Once you create the `ng_function`, you can use it to run computation on the Inference Engine.
As it was shown in [Build a Model with nGraph Library](nGraphTutorial.md), `std::shared_ptr<ngraph::Function>` can be transformed into a `CNNNetwork`.


### <a name="stream">Stream as Input</a>

The code below shows how to convert the ONNX ResNet50 model to the nGraph function using `import_onnx_model` with the stream as an input:

@snippet openvino/docs/snippets/OnnxImporterTutorial2.cpp part2

### <a name="path">Filepath as Input</a>

The code below shows how to convert the ONNX ResNet50 model to the nGraph function using `import_onnx_model` with the filepath as an input:

@snippet openvino/docs/snippets/OnnxImporterTutorial3.cpp part3

[onnx_header]: https://github.com/NervanaSystems/ngraph/blob/master/src/ngraph/frontend/onnx_import/onnx.hpp
[onnx_model_zoo]: https://github.com/onnx/models

