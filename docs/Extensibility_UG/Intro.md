# OpenVINO Extensibility Mechanism {#openvino_docs_Extensibility_UG_Intro}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_Extensibility_UG_add_openvino_ops
   openvino_docs_Extensibility_UG_Frontend_Extensions
   openvino_docs_Extensibility_UG_GPU
   openvino_docs_IE_DG_Extensibility_DG_VPU_Kernel
   openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer

@endsphinxdirective

The Intel® Distribution of OpenVINO™ toolkit supports neural network models trained with various frameworks, including
TensorFlow, PyTorch, ONNX, PaddlePaddle, MXNet, Caffe, and Kaldi. The list of supported operations is different for
each of the supported frameworks. To see the operations supported by your framework, refer to
[Supported Framework Operations](../MO_DG/prepare_model/Supported_Frameworks_Layers.md).

Custom operations, which are not included in the list, are not recognized by OpenVINO™ out-of-the-box. The need for custom operation may appear in two cases:

1. A new or rarely used regular framework operation is not supported in OpenVINO yet.

2. A new user operation that was created for some specific model topology by a model author using framework extension capabilities.

Importing models with such operations requires additional steps. This guide illustrates the workflow for running inference on models featuring custom operations, allowing you to plug in your own implementation for them. OpenVINO™ Extensibility API lets you add support for those custom operations and use one implementation for Model Optimizer and OpenVINO™ Runtime.

Defining a new custom operation basically consists of two parts:

1. Definition of operation semantics in OpenVINO, the code that describes how this operation should be inferred consuming input tensor(s) and producing output tensor(s). The implementation of execution kernels for [GPU](./GPU_Extensibility.md) and [VPU](./VPU_Extensibility.md) is described in separate guides.

2. Mapping rule that facilitates conversion of framework operation representation to OpenVINO defined operation semantics.

The first part is required for inference, the second part is required for successful import of a model containing such operations from the original framework model format. There are several options to implement each part. Next sections will describe them in detail.

## Definition of Operation Semantics

If the custom operation can be mathematically represented as a combination of exiting OpenVINO operations and such decomposition gives desired performance, then low-level operation implementation is not required. Refer to the latest OpenVINO operation set, when deciding feasibility of such decomposition. You can use any valid combination of exiting operations. The next section of this document describes the way to map a custom operation.

If such decomposition is not possible or appears too bulky with a large number of constituent operations that do not perform well, then a new class for the custom operation should be implemented as described in the [Custom Operation Guide](add_openvino_ops.md). 

You might prefer implementing a custom operation class if you already have a generic C++ implementation of operation kernel. Otherwise, try to decompose the operation first, as described above. Then, after verifying correctness of inference and resulting performance, you may move on to optional implementation of Bare Metal C++.

## Mapping from Framework Operation

Mapping of custom operation is implemented differently, depending on model format used for import. You may choose one of the following:

1. If model is represented in ONNX (including models exported from Pytorch in ONNX) or PaddlePaddle formats, then one of the classes from [Frontend Extension API](frontend_extensions.md) should be used. It consists of several classes available in C++ which can be used with Model Optimizer `--extensions` option or when model is imported directly to OpenVINO runtime using `read_model` method. Python API is also available for runtime model importing.

2. If model is represented in TensorFlow, Caffe, Kaldi or MXNet formats, then [Model Optimizer Extensions](../MO_DG/prepare_model/customize_model_optimizer/Customize_Model_Optimizer.md) should be used. This approach is available for model conversion in Model Optimizer only.

The simultaneous use of two approaches is explained by the two different types of frontends used for model conversion in OpenVINO: new frontends (ONNX, PaddlePaddle) and legacy frontends (TensorFlow, Caffe, Kaldi and MXNet). Model Optimizer can use both frontends in contrast to the direct import of model with `read_model` method which can use new frontends only. Follow one of the appropriate guides referenced above to implement mappings depending on framework frontend.

If you are implementing extensions for ONNX or PaddlePaddle new frontends and plan to use Model Optimizer `--extension` option for model conversion, then the extensions should be:

1. Implemented in C++ only

2. Compiled as a separate shared library (see details on how to do this further in this guide).

You cannot write new frontend extensions using Python API if you plan to use them with Model Optimizer.

Remaining part of this guide describes application of Frontend Extension API for new frontends.

## Registering Extensions

A custom operation class and a new mapping frontend extension class object should be registered to be usable in OpenVINO runtime.

> **NOTE**: This documentation is built upon the [Template extension](https://github.com/openvinotoolkit/openvino/tree/master/docs/template_extension/new), which demonstrates the details of extension development. It is based on minimalistic `Identity` operation that is a placeholder for your real custom operation. You may review the complete, fully compliable code to see how it works.

Use the `ov::Core::add_extension` method to load the extensions to the `ov::Core` object. This method allows to load library with extensions or extensions from the code.

### Load Extensions to Core

Extensions can be loaded from code with `ov::Core::add_extension` method:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_extensions.cpp add_extension

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_extensions.py add_extension

@endsphinxtab

@endsphinxtabset

`Identity` is a custom operation class defined in [Custom Operation Guide](add_openvino_ops.md). This is sufficient to enable reading IR which uses `Identity` extension operation emitted by Model Optimizer. In order to load original model directly to the runtime, you need to add also a mapping extension:

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_extensions.cpp
       :language: cpp
       :fragment: add_frontend_extension

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_extensions.py
       :language: python
       :fragment: add_frontend_extension

@endsphinxdirective
 
When Python API is used, there is no way to implement a custom OpenVINO operation. Also, even if custom OpenVINO operation is implemented in C++ and loaded into the runtime by a shared library, there is still no way to add a frontend mapping extension that refers to this custom operation. In this case, use C++ shared library approach to implement both operations semantics and framework mapping.

You may still use Python to map and decompose operations when only operations from the standard OpenVINO operation set are used.

### Create Library with Extensions

You need to create extension library in the following cases:
 - Convertion of model with custom operations in Model Optimizer
 - Loading model with custom operations in Python application. This applies to both framework model and IR.
 - Loading models with custom operations in tools that support loading extensions from a library, for example `benchmark_app`.

If you want to create an extension library, for example, to load these extensions into the Model Optimizer, you need to do the following:

Create an entry point for extension library. OpenVINO™ provides an `OPENVINO_CREATE_EXTENSIONS()` macro, which allows to define an entry point to a library with OpenVINO™ Extensions.
This macro should have a vector of all OpenVINO™ Extensions as an argument.

Based on that, the declaration of an extension class might look like the following:

@snippet template_extension/new/ov_extension.cpp ov_extension:entry_point

To configure the build of your extension library, use the following CMake script:

@snippet template_extension/new/CMakeLists.txt cmake:extension

This CMake script finds the OpenVINO™ using the `find_package` CMake command.

To build the extension library, run the commands below:

```sh
$ cd docs/template_extension/new
$ mkdir build
$ cd build
$ cmake -DOpenVINO_DIR=<OpenVINO_DIR> ../
$ cmake --build .
```

After the build, you may use path to your extension library to load your extensions to OpenVINO™ Runtime:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_extensions.cpp add_extension_lib

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_extensions.py add_extension_lib

@endsphinxtab

@endsphinxtabset

## See Also

* [OpenVINO Transformations](./ov_transformations.md)
* [Using OpenVINO Runtime Samples](../OV_Runtime_UG/Samples_Overview.md)
* [Hello Shape Infer SSD sample](../../samples/cpp/hello_reshape_ssd/README.md)