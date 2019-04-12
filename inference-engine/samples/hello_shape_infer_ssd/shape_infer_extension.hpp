// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <map>
#include <memory>
#include <string>
#include <algorithm>
#include <vector>

#include <inference_engine.hpp>

#define CUSTOM_RELU_TYPE std::string("CustomReLU")

class CustomReLUImpl : public InferenceEngine::ILayerExecImpl {
public:
    explicit CustomReLUImpl(const InferenceEngine::CNNLayer& layer) : _layer(layer) {}

    InferenceEngine::StatusCode getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig>& conf,
                                                           InferenceEngine::ResponseDesc* resp) noexcept override {
        InferenceEngine::DataConfig inDataConfig;
        InferenceEngine::DataConfig outDataConfig;
        auto firstInput = *_layer.insData.begin();
        auto firstOutput = *_layer.outData.begin();
        inDataConfig.desc = firstInput.lock()->getTensorDesc();
        outDataConfig.desc = firstOutput->getTensorDesc();
        InferenceEngine::LayerConfig layerConfig;
        layerConfig.inConfs = {inDataConfig};
        layerConfig.outConfs = {outDataConfig};
        conf.push_back(layerConfig);
        return InferenceEngine::StatusCode::OK;
    }

    InferenceEngine::StatusCode
    init(InferenceEngine::LayerConfig& config, InferenceEngine::ResponseDesc* resp) noexcept override {
        return InferenceEngine::StatusCode::OK;
    }

    InferenceEngine::StatusCode
    execute(std::vector<InferenceEngine::Blob::Ptr>& inputs, std::vector<InferenceEngine::Blob::Ptr>& outputs,
            InferenceEngine::ResponseDesc* resp) noexcept override {
        static bool wasCalled = false;
        if (!wasCalled) {
            std::cout << "Running " + CUSTOM_RELU_TYPE + " kernel for the first time (next messages won't be printed)"
                      << std::endl;
            wasCalled = true;
        }
        for (size_t i = 0; i < inputs.size(); i++) {
            auto inputBlob = inputs[i];
            auto outputBlob = outputs[i];
            auto inputData = inputBlob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
            auto outputData = outputBlob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
            for (size_t j = 0; j < inputBlob->size(); j++) {
                outputData[j] = inputData[j] < 0 ? 0 : inputData[j];
            }
        }
        return InferenceEngine::StatusCode::OK;
    }

private:
    const InferenceEngine::CNNLayer _layer;
};

class CustomReLUFactory : public InferenceEngine::ILayerImplFactory {
public:
    explicit CustomReLUFactory(const InferenceEngine::CNNLayer* layer) : _layer(*layer) {}

    InferenceEngine::StatusCode
    getImplementations(std::vector<InferenceEngine::ILayerImpl::Ptr>& impls,
                       InferenceEngine::ResponseDesc* resp) noexcept override {
        impls.push_back(std::make_shared<CustomReLUImpl>(_layer));
        return InferenceEngine::StatusCode::OK;
    }

private:
    InferenceEngine::CNNLayer _layer;
};

class CustomReLUResizeImpl : public InferenceEngine::IShapeInferImpl {
public:
    InferenceEngine::StatusCode inferShapes(const std::vector<InferenceEngine::Blob::CPtr>& inBlobs,
                                            const std::map<std::string, std::string>& params,
                                            const std::map<std::string, InferenceEngine::Blob::Ptr>& blobs,
                                            std::vector<InferenceEngine::SizeVector>& outShapes,
                                            InferenceEngine::ResponseDesc* desc) noexcept override {
        static bool wasCalled = false;
        if (!wasCalled) {
            std::cout << "Running " + CUSTOM_RELU_TYPE +
                         " shape inference for the first time (next messages won't be printed)" << std::endl;
            wasCalled = true;
        }
        for (const auto& blob : inBlobs) {
            outShapes.push_back(blob->getTensorDesc().getDims());
        }
        return InferenceEngine::StatusCode::OK;
    }
};

class InPlaceExtension : public InferenceEngine::IExtension {
public:
    InPlaceExtension() {
        _shapeInferImpl = std::make_shared<CustomReLUResizeImpl>();
    }

    InferenceEngine::StatusCode
    getPrimitiveTypes(char**& types, unsigned int& size, InferenceEngine::ResponseDesc* resp) noexcept override {
        size = 1;
        types = new char* [size];
        std::string type = CUSTOM_RELU_TYPE;
        types[0] = new char[type.size() + 1];
        std::copy(type.begin(), type.end(), types[0]);
        types[0][type.size()] = 0;
        return InferenceEngine::OK;
    };

    InferenceEngine::StatusCode
    getShapeInferTypes(char**& types, unsigned int& size, InferenceEngine::ResponseDesc* resp) noexcept override {
        return getPrimitiveTypes(types, size, resp);
    };

    InferenceEngine::StatusCode getShapeInferImpl(InferenceEngine::IShapeInferImpl::Ptr& impl, const char* type,
                                                  InferenceEngine::ResponseDesc* resp) noexcept override {
        if (CUSTOM_RELU_TYPE.compare(type) != 0) return InferenceEngine::StatusCode::NOT_IMPLEMENTED;
        impl = _shapeInferImpl;
        return InferenceEngine::StatusCode::OK;
    }

    void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept override {};

    void SetLogCallback(InferenceEngine::IErrorListener& listener) noexcept override {};

    void Unload() noexcept override {};

    void Release() noexcept override {}

    InferenceEngine::StatusCode
    getFactoryFor(InferenceEngine::ILayerImplFactory*& factory, const InferenceEngine::CNNLayer* cnnLayer,
                  InferenceEngine::ResponseDesc* resp) noexcept override {
        if (cnnLayer->type != CUSTOM_RELU_TYPE)
            return InferenceEngine::StatusCode::NOT_IMPLEMENTED;
        factory = new CustomReLUFactory(cnnLayer);
        return InferenceEngine::StatusCode::OK;
    };

private:
    InferenceEngine::IShapeInferImpl::Ptr _shapeInferImpl;
};
