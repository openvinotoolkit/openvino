# Old ShapeInference Extensibility API {#openvino_docs_IE_DG_Extensibility_DG_deprecated_ShapeInfer}

The new approach to shape inference suggests a creation of a custom nGraph operation that contains a special method for shape inference. 
The following classes and methods were deprecated:

 * `InferenceEngine::IShapeInferExtension` class
 * `InferenceEngine::IShapeInferExtension::getShapeInferTypes(char**&, unsigned int&, ResponseDesc*)` method
 * `InferenceEngine::IShapeInferExtension::getShapeInferImpl(IShapeInferImpl::Ptr&, const char*, ResponseDesc*)` method

However, the old approach with the `InferenceEngine::IShapeInferExtension` method still works for already existing custom layers.
Custom Shape Inference functions are registered by calling `InferenceEngine::ICNNNetwork::AddExtension` with the implemented `InferenceEngine::IShapeInferExtension` method, which is a holder of custom implementations. 
The holder requires to implement two key methods:
* `InferenceEngine::IShapeInferExtension::getShapeInferImpl` - Returns custom shape inference implementation for the given type. 
* `InferenceEngine::IShapeInferExtension::getShapeInferTypes` - Provides all custom types.

Custom shape inference implementation is represented by the `InferenceEngine::IShapeInferImpl::inferShapes` method.

It is impossible to overwrite built-in shape inference functions. Custom type must be different from the supported ones. 
