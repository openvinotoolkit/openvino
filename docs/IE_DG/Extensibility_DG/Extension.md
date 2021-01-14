# Extension Library {#openvino_docs_IE_DG_Extensibility_DG_Extension}

Inference Engine provides an InferenceEngine::IExtension interface, which defines the interface for Inference Engine Extension libraries.
All extension libraries should be inherited from this interface. The example below contains implementation of two operations: `Template`
used as an example in this document and `FFT` used as a more complex example from the [Custom Operations Guide](../../HOWTO/Custom_Layers_Guide.md).

> **NOTE**: `FFT` operation is implemented using OpenCV library functions `cv::dft` and `cv::idft`.

Based on that, declaration of an extension class can look as follows:

@snippet template_extension/extension.hpp extension:header

The extension library should contain and export the method InferenceEngine::CreateExtension, which creates an `Extension` class:

@snippet template_extension/extension.cpp extension:CreateExtension

Also, an `Extension` object should implement the following methods:

* InferenceEngine::IExtension::Release deletes an extension object

* InferenceEngine::IExtension::GetVersion returns information about version of the library

@snippet template_extension/extension.cpp extension:GetVersion

Implement the  InferenceEngine::IExtension::getOpSets method if the extension contains custom layers. 
Read the [guide about custom operations](AddingNGraphOps.md) for more information.

To understand how integrate execution kernels to the extension library, read the [guide about development of custom CPU kernels](CPU_Kernel.md).
To understand how to register custom ONNX operator to the extension library, read the [guide about custom ONNX operators](Custom_ONNX_Ops.md).
