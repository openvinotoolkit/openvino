# Inference Engine Extensibility Mechanism {#openvino_docs_IE_DG_Extensibility_DG_Intro}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:
   
   openvino_docs_IE_DG_Extensibility_DG_AddingNGraphOps
   openvino_docs_IE_DG_Extensibility_DG_Custom_ONNX_Ops
   CPU Kernels Extensibility <openvino_docs_IE_DG_Extensibility_DG_CPU_Kernel>
   GPU Kernels Extensibility <openvino_docs_IE_DG_Extensibility_DG_GPU_Kernel>
   VPU Kernels Extensibility <openvino_docs_IE_DG_Extensibility_DG_VPU_Kernel>
   openvino_docs_IE_DG_Extensibility_DG_Extension
   openvino_docs_IE_DG_Extensibility_DG_Building

@endsphinxdirective

If your model contains operations not normally supported by OpenVINO, the Inference Engine Extensibility API lets you add support for those custom operations in a library containing custom nGraph operation sets, corresponding extensions to the Model Optimizer, and a device plugin extension. See the overview in the [Custom Operations Guide](../../HOWTO/Custom_Layers_Guide.md) to learn how these work together.

To load the Extensibility library to the `InferenceEngine::Core` object, use the `InferenceEngine::Core::AddExtension` method.

## Inference Engine Extension Library

An Inference Engine Extension dynamic library contains the following components:

 * [Extension Library](Extension.md):
    - Contains custom operation sets
    - Provides CPU implementations for custom operations
 * [Custom nGraph Operation](AddingNGraphOps.md):
    - Enables the use of `InferenceEngine::Core::ReadNetwork` to read Intermediate Representation (IR) with unsupported
    operations
    - Enables the creation of `ngraph::Function` with unsupported operations
    - Provides a shape inference mechanism for custom operations

> **NOTE**: This documentation is written based on the [Template extension](https://github.com/openvinotoolkit/openvino/tree/master/docs/template_extension), which demonstrates extension development details. You can review the complete code, which is fully compilable and up-to-date, to see how it works.

## Execution Kernels

The Inference Engine workflow involves the creation of custom kernels and either custom or existing operations.

An _operation_ is a network building block implemented in the training framework, for example, `Convolution` in Caffe*.
A _kernel_ is defined as the corresponding implementation in the Inference Engine.

Refer to the [Model Optimizer Extensibility](../../MO_DG/prepare_model/customize_model_optimizer/Customize_Model_Optimizer.md)
for details on how a mapping between framework operations and Inference Engine kernels is registered.

In short, you can plug your own kernel implementations into the Inference Engine and map them to the operations in the original framework.

The following pages describe how to integrate custom _kernels_ into the Inference Engine:

 * [Introduction to development of custom CPU kernels](CPU_Kernel.md)
 * [Introduction to development of custom GPU kernels](GPU_Kernel.md)
 * [Introduction to development of custom VPU kernels](VPU_Kernel.md)

## See Also

* [Build an extension library using CMake*](Building.md)
* [Using Inference Engine Samples](../Samples_Overview.md)
* [Hello Shape Infer SSD sample](../../../samples/cpp/hello_reshape_ssd/README.md)
