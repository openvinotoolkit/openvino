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
```cpp
const std::int64_t version = 12;
const std::string domain = "ai.onnx";
const std::set<std::string> supported_ops = ngraph::onnx_import::get_supported_operators(version, domain);

for(const auto& op : supported_ops)
{
    std::cout << op << std::endl;
}
```
The above code produces a list of all the supported operators for the `version` and `domain` you specified and outputs a list similar to this:
```cpp
Abs
Acos
...
Xor
```

To determine whether a specific ONNX operator in a particular version and domain is supported by the importer, use the `is_operator_supported` function as shown in the example below:
```cpp
const std::string op_name = "Abs";
const std::int64_t version = 12;
const std::string domain = "ai.onnx";
const bool is_abs_op_supported = ngraph::onnx_import::is_operator_supported(op_name, version, domain);

std::cout << "Abs in version 12, domain `ai.onnx`is supported: " << (is_abs_op_supported ? "true" : "false") << std::endl;
```

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

```cpp
 const std::string resnet50_path = "resnet50/model.onnx";
 std::ifstream resnet50_stream(resnet50_path);
 if(resnet50_stream.is_open())
 {
     try
     {
         const std::shared_ptr<ngraph::Function> ng_function = ngraph::onnx_import::import_onnx_model(resnet50_stream);

         // Check shape of the first output, for example
         std::cout << ng_function->get_output_shape(0) << std::endl;
         // The output is Shape{1, 1000}
     }
     catch (const ngraph::ngraph_error& error)
     {
         std::cout << "Error when importing ONNX model: " << error.what() << std::endl;
     }
 }
 resnet50_stream.close();
```

### <a name="path">Filepath as Input</a>

The code below shows how to convert the ONNX ResNet50 model to the nGraph function using `import_onnx_model` with the filepath as an input:
```cpp
const std::shared_ptr<ngraph::Function> ng_function = ngraph::onnx_import::import_onnx_model(resnet50_path);
```

[onnx_header]: https://github.com/NervanaSystems/ngraph/blob/master/src/ngraph/frontend/onnx_import/onnx.hpp
[onnx_model_zoo]: https://github.com/onnx/models


## Deprecation Notice

<table>
  <tr>
    <td><strong>Deprecation Begins</strong></td>
    <td>June 1, 2020</td>
  </tr>
  <tr>
    <td><strong>Removal Date</strong></td>
    <td>December 1, 2020</td>
  </tr>
</table> 

*Starting with the OpenVINO™ toolkit 2020.2 release, all of the features previously available through nGraph have been merged into the OpenVINO™ toolkit. As a result, all the features previously available through ONNX RT Execution Provider for nGraph have been merged with ONNX RT Execution Provider for OpenVINO™ toolkit.*

*Therefore, ONNX RT Execution Provider for nGraph will be deprecated starting June 1, 2020 and will be completely removed on December 1, 2020. Users are recommended to migrate to the ONNX RT Execution Provider for OpenVINO™ toolkit as the unified solution for all AI inferencing on Intel® hardware.*