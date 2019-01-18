// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <details/caseless.hpp>
#include <ie_inetwork.hpp>
#include <ie_layers.h>
#include <ie_blob.h>
#include <memory>
#include <string>

namespace InferenceEngine {

namespace Builder {

class BaseConverter {
public:
    explicit BaseConverter(const std::string& type): type(type) {}

    virtual CNNLayer::Ptr createLayer(const std::shared_ptr<const ILayer>& layer, Precision precision) = 0;
    virtual bool canCreate(const std::string& nodeType) const = 0;

protected:
    std::string type;
};

template <class CLT>
class LayerConverter: public BaseConverter {
public:
    explicit LayerConverter(const std::string& type): BaseConverter(type) {}

    CNNLayer::Ptr createLayer(const std::shared_ptr<const ILayer>& layer, Precision precision) override {
        LayerParams params = {layer->getName(), layer->getType(), precision};
        auto res = std::make_shared<CLT>(params);

        auto * weightLayerPtr = dynamic_cast<WeightableLayer *>(res.get());

        for (auto& it : layer->getParameters()->getConstantData()) {
            res->blobs[it.first] = std::const_pointer_cast<Blob>(it.second);
            if (weightLayerPtr == nullptr)
                continue;
            if (it.first == "weights") {
                weightLayerPtr->_weights =  std::const_pointer_cast<Blob>(it.second);
            } else if (it.first == "biases") {
                weightLayerPtr->_biases =  std::const_pointer_cast<Blob>(it.second);
            }
        }

        for (const auto& it : layer->getParameters()->getParameters()) {
            res->params[it.first] = it.second;
        }
        return res;
    }

    bool canCreate(const std::string& nodeType) const override {
        details::CaselessEq<std::string> comparator;
        return comparator(nodeType, type);
    }
};

class ActivationConverter: public BaseConverter {
public:
    ActivationConverter(): BaseConverter("Activation") {}

    CNNLayer::Ptr createLayer(const std::shared_ptr<const ILayer>& layer, Precision precision) override {
        LayerParams params = {layer->getName(), layer->getType(), precision};
        static details::caseless_map<std::string, std::shared_ptr<BaseConverter>> activationCreators = {
                {"relu", std::make_shared<LayerConverter<InferenceEngine::ReLULayer>>("ReLU")},
                {"prelu", std::make_shared<LayerConverter<InferenceEngine::PReLULayer>>("PReLU")},
                {"clamp", std::make_shared<LayerConverter<InferenceEngine::ClampLayer>>("Clamp")},
                {"elu", std::make_shared<LayerConverter<InferenceEngine::CNNLayer>>("ELU")},
                {"sigmoid", std::make_shared<LayerConverter<InferenceEngine::CNNLayer>>("Sigmoid")},
                {"tanh", std::make_shared<LayerConverter<InferenceEngine::CNNLayer>>("TanH")},
        };

        auto typeIt = layer->getParameters()->getParameters().find("type");
        if (typeIt == layer->getParameters()->getParameters().end())
            THROW_IE_EXCEPTION << "Unsupported Activation layer. Type is unknown.";

        auto activationBuilder = activationCreators.find(typeIt->second);
        if (activationBuilder == activationCreators.end()) {
            THROW_IE_EXCEPTION << "Unsupported Activation layer type: " << typeIt->second.asString();
        }

        auto activation = activationBuilder->second->createLayer(layer, precision);

        activation->type = activationBuilder->first;
        activation->params.erase("type");
        activation->validateLayer();
        return activation;
    }

    bool canCreate(const std::string& nodeType) const override {
        details::CaselessEq<std::string> comparator;
        return comparator(nodeType, type);
    }
};

}  // namespace Builder
}  // namespace InferenceEngine
