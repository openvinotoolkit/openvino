# OpenVINO Extensibility Mechanism {#openvino_docs_Extensibility_UG_Intro}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_Extensibility_UG_add_openvino_ops

@endsphinxdirective

The Intel® Distribution of OpenVINO™ toolkit supports neural network models trained with multiple frameworks including
TensorFlow, Caffe, MXNet, Kaldi, PaddlePaddle, and ONNX. The list of supported operations (layers) is different for
each of the supported frameworks. To see the operations supported by your framework, refer to
[Supported Framework Operations](../MO_DG/prepare_model/Supported_Frameworks_Layers.md).

Custom operations, that is those not included in the list, are not recognized by OpenVINO™ out-of-the-box. Therefore, creating Intermediate Representation (IR) for a model using them requires additional steps. This guide illustrates the workflow for running inference on topologies featuring custom operations, allowing you to plug in your own implementation for existing or completely new operations.

If your model contains operations not normally supported by OpenVINO™, the OpenVINO™ Extensibility API lets you add support for those custom operations and use one implementation for Model Optimizer and OpenVINO™ Runtime.

There are two steps to support inference of a model with custom operation(s):
1. Add support for a [custom operation in the Model Optimizer](../MO_DG/prepare_model/customize_model_optimizer/Customize_Model_Optimizer.md) so
the Model Optimizer can generate the IR with the operation.
2. Create a custom operation in it as described in the [Custom Operation](add_openvino_ops.md).

## OpenVINO™ Extensions

An OpenVINO™ provides extensions for:

 * [Custom OpenVINO™ Operation](add_openvino_ops.md):
    - Enables the creation of unsupported operations
    - Enables the use of `ov::Core::read_model` to read models with unsupported operations
    - Provides a shape inference mechanism for custom operations
    - Provides an evaluate method which allow to support the operation on CPU or perform constant folding

> **NOTE**: This documentation is written based on the [Template extension](https://github.com/openvinotoolkit/openvino/tree/master/docs/template_extension/new), which demonstrates extension development details. You can review the complete code, which is fully compilable and up-to-date, to see how it works.

## Load extensions to OpenVINO™ Runtime

To load the extensions to the `ov::Core` object, use the `ov::Core::add_extension` method, this method allows to load library with extensions or extensions from the code.

### Load extensions to core

Extensions can be loaded from code with `ov::Core::add_extension` method:

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_extensions.cpp
       :language: cpp
       :fragment: add_extension

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_extensions.py
       :language: python
       :fragment: add_extension

@endsphinxdirective

### Create library with extensions

You need to create extension library in folowing cases:
 - Load extensions to Model Optimizer
 - Load extensions to Python application

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
$ cd docs/template_extension/new
$ mkdir build
$ cd build
$ cmake -DOpenVINO_DIR=<OpenVINO_DIR> ../
$ cmake --build .
```

After the build you can use path to your extension library to load your extensions to OpenVINO™ Runtime:

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_extensions.cpp
       :language: cpp
       :fragment: add_extension_lib

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_extensions.py
       :language: python
       :fragment: add_extension_lib

@endsphinxdirective

## See Also

* [Using Inference Engine Samples](../OV_Runtime_UG/Samples_Overview.md)
* [Hello Shape Infer SSD sample](../../samples/cpp/hello_reshape_ssd/README.md)
