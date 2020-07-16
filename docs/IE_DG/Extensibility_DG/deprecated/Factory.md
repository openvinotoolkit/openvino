# Deprecated API for CPU kernels creation {#openvino_docs_IE_DG_Extensibility_DG_deprecated_Factory}

List of deprecated API for kernels development:
 * `InferenceEngine::IExtension::getPrimitiveTypes(char**& types, unsigned int& size, ResponseDesc* resp)` method
 * `InferenceEngine::IExtension::getFactoryFor(ILayerImplFactory *&factory, const CNNLayer *cnnLayer, ResponseDesc *resp)` method
 * `InferenceEngine::ILayerImplFactory` class

>**NOTE**: This guide demonstrates how to use deprecated API for kernels creation. However, keep in mind that this API will be deleted soon.

1. Create your custom layer factory `CustomLayerFactory` class:
```cpp
// custom_layer.h
// A CustomLayerFactory class is an example layer, which makes exponentiation by 2 for the input and does not change dimensions
class CustomLayerFactory {

};
```
2. Inherit it from the abstract `InferenceEngine::ILayerImplFactory` class: 
```cpp
// custom_layer.h
class CustomLayerFactory: public InferenceEngine::ILayerImplFactory {

};
```

3. Create a constructor, a virtual destructor, and a data member to keep the layer info:
```cpp
// custom_layer.h
class CustomLayerFactory: public InferenceEngine::ILayerImplFactory {
public:
    explicit CustomLayerFactory(const CNNLayer *layer): cnnLayer(*layer) {}
private:
  CNNLayer cnnLayer;
};
```

4. Overload and implement the abstract methods `getShapes` and `getImplementations` of the `InferenceEngine::ILayerImplFactory` class:
```cpp
// custom_layer.h
class CustomLayerFactory: public InferenceEngine::ILayerImplFactory {
public:
    // ... constructor and destructor

    StatusCode getShapes(const std::vector<TensorDesc>& inShapes, std::vector<TensorDesc>& outShapes, ResponseDesc *resp) noexcept override {
        if (cnnLayer == nullptr) {
            std::string errorMsg = "Cannot get cnn layer!";
            errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            return GENERAL_ERROR;
        }
        if (inShapes.size() != 1) {
            std::string errorMsg = "Incorrect input shapes!";
            errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            return GENERAL_ERROR;
        }
        outShapes.clear();
        outShapes.emplace_back(inShapes[0]);
        return OK;
    }

    StatusCode getImplementations(std::vector<ILayerImpl::Ptr>& impls, ResponseDesc *resp) noexcept override {
        // You can add cnnLayer to implementation if it is necessary
        impls.push_back(ILayerImpl::Ptr(new CustomLayerImpl()));
        return OK;
    }
};
```
5. Create your custom layer implementation `CustomLayerImpl` class using the [instruction](../CPU_Kernel.md).

6. Implement methods in the `Extension` class:
```cpp
// custom_extension.h
class CustomExtention : public InferenceEngine::IExtension {
public:
    // ... utility methods
    // Retruns the list of supported kernels/layers
    StatusCode getPrimitiveTypes(char**& types, unsigned int& size, ResponseDesc* resp) noexcept override {
        std::string type_name = "CustomLayer";
        types = new char *[1];
        size = 1;
        types[0] = new char[type_name.size() + 1];
        std::copy(type_name.begin(), type_name.end(), types[0]);
        types[0][type_name.size()] = '\0';
        return OK;
    }
    // Main function
    StatusCode getFactoryFor(ILayerImplFactory *&factory, const CNNLayer *cnnLayer, ResponseDesc *resp) noexcept override {
        if (cnnLayer->type != "CustomLayer") {
            std::string errorMsg = std::string("Factory for ") + cnnLayer->type + " wasn't found!";
            errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            return NOT_FOUND;
        }
        factory = new CustomLayerFactory(cnnLayer);
        return OK;
    }
};
```
