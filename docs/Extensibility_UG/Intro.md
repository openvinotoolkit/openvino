# OpenVINO Extensibility Mechanism {#openvino_docs_Extensibility_UG_Intro}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_Extensibility_UG_add_openvino_ops
   openvino_docs_Extensibility_UG_Extension
   openvino_docs_Extensibility_UG_Building

@endsphinxdirective

The Intel® Distribution of OpenVINO™ toolkit supports neural network models trained with multiple frameworks including
TensorFlow*, Caffe*, MXNet*, Kaldi* and ONNX* file format. The list of supported operations (layers) is different for
each of the supported frameworks. To see the operations supported by your framework, refer to
[Supported Framework Layers](../MO_DG/prepare_model/Supported_Frameworks_Layers.md).

Custom operations, that is those not included in the list, are not recognized by Model Optimizer out-of-the-box. Therefore, creating Intermediate Representation (IR) for a model using them requires additional steps. This guide illustrates the workflow for running inference on topologies featuring custom operations, allowing you to plug in your own implementation for existing or completely new operations.

> **NOTE**: *Layer* is a legacy term for *operation* which came from Caffe\* framework. Currently it is not used.
> Refer to the [Deep Learning Network Intermediate Representation and Operation Sets in OpenVINO™](../MO_DG/IR_and_opsets.md)
> for more information on the topic.

If your model contains operations not normally supported by OpenVINO™, the OpenVINO™ Extensibility API lets you add support for those custom operations in a library containing custom operation and use one implementation for Model Optimizer and OpenVINO™ Runtime.

There are three steps to support inference of a model with custom operation(s):
1. Add support for a [custom operation in the Model Optimizer](../MO_DG/prepare_model/customize_model_optimizer/Customize_Model_Optimizer.md) so
the Model Optimizer can generate the IR with the operation.
2. Create a custom operation in it as described in the [Custom Operation](add_openvino_ops.md).

To load the Extensibility library to the `ov::Core` object, use the `ov::Core::add_extension` method.

## OpenVINO™ Extension Library

An OpenVINO™ Extension dynamic library contains the following components:

 * [Extension Library](Extension.md):
    - Contains custom operation
 * [Custom OpenVINO™ Operation](add_openvino_ops):
    - Enables the use of `ov::Core::read_model` to read Intermediate Representation (IR) with unsupported operations
    - Enables the creation of `ov::Model` with unsupported operations
    - Provides a shape inference mechanism for custom operations
    - Provides an evaluate method which allow to support the operation on CPU or perform constant folding

> **NOTE**: This documentation is written based on the [Template extension](https://github.com/openvinotoolkit/openvino/tree/master/docs/template_extension/new), which demonstrates extension development details. You can review the complete code, which is fully compilable and up-to-date, to see how it works.

## Execution Kernels

The OpenVINO™ workflow involves the creation of custom kernels and either custom or existing operations.

An _operation_ is a network building block implemented in the training framework, for example, `Convolution` in Caffe*.
A _kernel_ is defined as the corresponding implementation in the OpenVINO™.

Refer to the [Model Optimizer Extensibility](../MO_DG/prepare_model/customize_model_optimizer/Customize_Model_Optimizer.md)
for details on how a mapping between framework operations and OpenVINO™ operations.

In short, you can plug your own kernel implementations into the OpenVINO™ and map them to the operations in the original framework.

## See Also

* [Build an extension library using CMake*](Building.md)
* [Using Inference Engine Samples](../OV_Runtime_UG/Samples_Overview.md)
* [Hello Shape Infer SSD sample](../../samples/cpp/hello_reshape_ssd/README.md)
