# OpenVINO Extensibility Mechanism {#openvino_docs_Extensibility_UG_Intro}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_Extensibility_UG_add_openvino_ops
   openvino_docs_Extensibility_UG_Frontend_Extensions
   openvino_docs_Extensibility_UG_GPU
   openvino_docs_Extensibility_UG_VPU_Kernel
   openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer

@endsphinxdirective

The Intel® Distribution of OpenVINO™ toolkit supports neural network models trained with various frameworks, including
TensorFlow, PyTorch, ONNX, PaddlePaddle, Apache MXNet, Caffe, and Kaldi. The list of supported operations is different for
each of the supported frameworks. To see the operations supported by your framework, refer to
[Supported Framework Operations](../MO_DG/prepare_model/Supported_Frameworks_Layers.md).

Custom operations, that is those not included in the list, are not recognized by OpenVINO™ out-of-the-box. The need for a custom operation may appear in two main cases:

1. A regular framework operation that is new or rarely used, which is why it hasn’t been implemented in OpenVINO yet.

2. A new user operation that was created for some specific model topology by a model author using framework extension capabilities.

Importing models with such operations requires additional steps. This guide illustrates the workflow for running inference on models featuring custom operations, allowing you to plug in your own implementation for them. OpenVINO™ Extensibility API lets you add support for those custom operations and use one implementation for Model Optimizer and OpenVINO™ Runtime.

Defining a new custom operation basically consist of two parts:

1. Definition of operation semantics in OpenVINO, the code that describes how this operation should be inferred consuming input tensor(s) and producing output tensor(s). How to implement execution kernels for [GPU](./GPU_Extensibility.md) and [VPU](./VPU_Extensibility.md) is described in separate guides.

2. Mapping rule that facilitates conversion of framework operation representation to OpenVINO defined operation semantics.

The first part is required for inference, the second part is required for successful import of a model containing such operations from the original framework model format. There are several options to implement each part, the next sections will describe them in detail.

## Definition of Operation Semantics


If the custom operation can be mathematically represented as a combination of exiting OpenVINO operations and such decomposition gives desired performance, then low-level operation implementation is not required. When deciding feasibility of such decomposition refer to the latest OpenVINO operation set. You can use any valid combination of exiting operations. How to map a custom operation is described in the next section of this document.

If such decomposition is not possible or appears too bulky with lots of consisting operations that are not performing well, then a new class for the custom operation should be implemented as described in the [Custom Operation Guide](add_openvino_ops.md). 

Prefer implementing a custom operation class if you already have a generic C++ implementation of operation kernel. Otherwise try to decompose the operation first as described above and then after verifying correctness of inference and resulting performance, optionally invest to implementing bare metal C++ implementation.

## Mapping from Framework Operation

Depending on model format used for import, mapping of custom operation is implemented differently, choose one of:

1. If model is represented in ONNX (including models exported from Pytorch in ONNX) or PaddlePaddle formats, then one of the classes from [Frontend Extension API](frontend_extensions.md) should be used. It consists of several classes available in C++ which can be used with Model Optimizer `--extensions` option or when model is imported directly to OpenVINO run-time using read_model method. Python API is also available for run-time model importing.

2. If model is represented in TensorFlow, Caffe, Kaldi or MXNet formats, then [Model Optimizer Extensions](../MO_DG/prepare_model/customize_model_optimizer/Customize_Model_Optimizer.md) should be used. This approach is available for model conversion in Model Optimizer only.

Existing of two approaches simultaneously is explained by two different types of frontends used for model conversion in OpenVINO: new frontends (ONNX, PaddlePaddle) and legacy frontends (TensorFlow, Caffe, Kaldi and Apache MXNet). Model Optimizer can use both front-ends in contrast to the direct import of model with `read_model` method which can use new frontends only. Follow one of the appropriate guides referenced above to implement mappings depending on framework frontend.

If you are implementing extensions for ONNX or PaddlePaddle new frontends and plan to use Model Optimizer `--extension` option for model conversion, then the extensions should be

1. Implemented in C++ only

2. Compiled as a separate shared library (see details how to do that later in this guide).

You cannot write new frontend extensions using Python API if you plan to use them with Model Optimizer.

Remaining part of this guide uses Frontend Extension API applicable for new frontends.

## Registering Extensions

A custom operation class and a new mapping frontend extension class object should be registered to be usable in OpenVINO runtime.

> **NOTE**: This documentation is written based on the [Template extension](https://github.com/openvinotoolkit/openvino/tree/master/src/core/template_extension/new), which demonstrates extension development details based on minimalistic `Identity` operation that is a placeholder for your real custom operation. You can review the complete code, which is fully compliable, to see how it works.

To load the extensions to the `ov::Core` object, use the `ov::Core::add_extension` method, this method allows to load library with extensions or extensions from the code.

### Load extensions to core

Extensions can be loaded from code with `ov::Core::add_extension` method:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_extensions.cpp add_extension

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_extensions.py add_extension

@endsphinxtab

@endsphinxtabset

`Identity` is custom operation class defined in [Custom Operation Guide](add_openvino_ops.md). This is enough to enable reading IR which uses `Identity` extension operation emitted by Model Optimizer. To be able to load original model directly to the runtime, you need to add also a mapping extension:

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
 
When Python API is used there is no way to implement a custom OpenVINO operation. Also, even if custom OpenVINO operation is implemented in C++ and loaded to the runtime through a shared library, there is still no way to add a frontend mapping extension that refers to this custom operation. Use C++ shared library approach to implement both operations semantics and framework mapping in this case.

You still can use Python for operation mapping and decomposition in case if operations from the standard OpenVINO operation set is used only.

### Create library with extensions

You need to create extension library in the following cases:
 - Convert model with custom operations in Model Optimizer
 - Load model with custom operations in Python application. It is applicable for both framework model and IR.
 - Loading models with custom operations in tools that support loading extensions from a library, for example `benchmark_app`.

If you want to create an extension library, for example in order to load these extensions to the Model Optimizer, you need to do next steps:
Create an entry point for extension library. OpenVINO™ provides an `OPENVINO_CREATE_EXTENSIONS()` macro, which allows to define an entry point to a library with OpenVINO™ Extensions.
This macro should have a vector of all OpenVINO™ Extensions as an argument.

Based on that, the declaration of an extension class can look as follows:

@snippet template_extension/new/ov_extension.cpp ov_extension:entry_point

To configure the build of your extension library, use the following CMake script:

@snippet template_extension/new/CMakeLists.txt cmake:extension

This CMake script finds the OpenVINO™ using the `find_package` CMake command.

To build the extension library, run the commands below:

```sh
$ cd src/core/template_extension/new
$ mkdir build
$ cd build
$ cmake -DOpenVINO_DIR=<OpenVINO_DIR> ../
$ cmake --build .
```

After the build you can use path to your extension library to load your extensions to OpenVINO™ Runtime:

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

