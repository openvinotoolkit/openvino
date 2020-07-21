# Extension Library {#openvino_docs_IE_DG_Extensibility_DG_Extension}

Inference Engine provides an InferenceEngine::IExtension interface, which defines the interface for Inference Engine Extension libraries.
All extension libraries should be inherited from this interface.

Based on that, declaration of an extension class can look as follows:

@snippet extension.hpp extension:header

The extension library should contain and export the method InferenceEngine::CreateExtension, which creates an `Extension` class:

@snippet extension.cpp extension:CreateExtension

Also, an `Extension` object should implement the following methods:

* InferenceEngine::IExtension::Release deletes an extension object

* InferenceEngine::IExtension::GetVersion returns information about version of the library

@snippet extension.cpp extension:GetVersion

Implement the  InferenceEngine::IExtension::getOpSets method if the extension contains custom layers. 
Read the [guide about custom operations](AddingNGraphOps.md) for more information.

To understand how integrate execution kernels to the extension library, read the [guide about development of custom CPU kernels](CPU_Kernel.md).
