# Inference Engine Extensibility Mechanism {#openvino_docs_IE_DG_Extensibility_DG_Intro}

Inference Engine Extensibility API allows to add support of custom operations to the Inference Engine.
Extension should contain operation sets with custom operations and execution kernels for custom operations.
Physically, an extension library can be represented as a dynamic library exporting the single `CreateExtension` function that allows to create a new extension instance.

Extensibility library can be loaded to the InferenceEngine::Core object using the InferenceEngine::Core::AddExtension method.

## Inference Engine Extension Library

Inference Engine Extension dynamic library contains several main components:

 * [Extension class](Extension.md):
    - Contains custom operation sets
    - Provides CPU implementations for custom operations
 * [Custom operations](Intro.md):
    - Allows to use InferenceEngine::Core::ReadNetwork to read Intermediate Representation (IR) with unsupported operations 
    - Allows to create `ngraph::Function` with unsupported operations
    - Provides shape inference mechanism for custom operations

> **NOTE**: This documentation is written based on the `Template extension`, which demonstrates extension 
development details. Find the complete code of the `Template extension`, which is fully compilable and up-to-date,
at `<dldt source tree>/docs/template_extension`.

## Execution Kernels

The Inference Engine workflow involves the creation of custom kernels and either custom or existing operations.

An _Operation_ is a Network building block implemented in the training framework, for example, `Convolution` in Caffe*.
A _Kernel_ is defined as the corresponding implementation in the Inference Engine.

Refer to the [Custom Layers in the Model Optimizer](../../MO_DG/prepare_model/customize_model_optimizer/Customize_Model_Optimizer.md) section for details on how
mapping between framework layers and Inference Engine kernels is registered.

In short, you can plug your own kernel implementations into the Inference Engine and map them to the layers in the original framework.

The following pages describe how to integrate custom _kernels_ into the Inference Engine:

 * [Introduction to development of custom CPU kernels](CPU_Kernel.md)
 * [Introduction to development of custom GPU kernels](GPU_Kernel.md)
 * [Introduction to development of custom VPU kernels](VPU_Kernel.md)

## Deprecated Extensibility API

Shape Inference API and some methods of extensibility mechanism was deprecated and will be removed soon.
Old Extensibility mechanism contains two parts shape inference and execution kernel.
 * [Shape Inference](deprecated/ShapeInfer.md)
 * [Execution Kernel](deprecated/Factory.md)

## Additional Resources

* [Build an extension library using CMake*](Building.md)

## See Also
* [Using Inference Engine Samples](../Samples_Overview.md)
* [Hello Shape Infer SSD sample](../../../inference-engine/samples/hello_reshape_ssd/README.md)
