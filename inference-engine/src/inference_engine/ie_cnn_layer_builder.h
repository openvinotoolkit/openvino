// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <details/caseless.hpp>
#include <ie_network.hpp>
#include <ie_builders.hpp>
#include <ie_layers.h>
#include <ie_blob.h>
#include <memory>
#include <string>
#include <vector>
#include <map>

#include <ngraph/node.hpp>

namespace InferenceEngine {

namespace Builder {

template<class T>
inline std::string convertParameter2String(const Parameter& parameter) {
    if (parameter.is<std::vector<T>>()) {
        std::vector<T> params = parameter.as<std::vector<T>>();
        std::string result;
        for (const auto& param : params) {
            if (!result.empty())
                result += ",";
            result += convertParameter2String<T>(param);
        }
        return result;
    }
    return std::to_string(parameter.as<T>());
}
template<>
inline std::string convertParameter2String<std::string>(const Parameter& parameter) {
    return parameter.as<std::string>();
}

std::map<std::string, std::string> convertParameters2Strings(const std::map<std::string, Parameter>& parameters);
Layer builderFromCNNLayer(const CNNLayerPtr& cnnLayer);

struct ConvertersHolder {
    details::caseless_map<std::string, std::function<void(const CNNLayerPtr& cnnLayer, Layer&)>> converters;
};

/**
 * @brief This class registers layer validators
 */
class ConverterRegister {
public:
    /**
     * @brief The constructor registers new layer validator
     * @param type Layer type
     * @param validator Layer validator
     */
    explicit ConverterRegister(const std::string& type, const std::function<void(const CNNLayerPtr&, Layer&)>& converter);

    static void convert(const CNNLayerPtr& cnnLayer, Layer& layer) {
        if (getConvertersHolder().converters.find(layer.getType()) != getConvertersHolder().converters.end())
            getConvertersHolder().converters[layer.getType()](cnnLayer, layer);
    }

private:
    static ConvertersHolder& getConvertersHolder();
};

#define REG_CONVERTER_FOR(__type, __converter) \
static InferenceEngine::Builder::ConverterRegister _reg_converter_##__type(#__type, __converter)

class INodeConverter {
public:
    virtual ~INodeConverter() = default;
    virtual CNNLayer::Ptr createLayer(const std::shared_ptr<ngraph::Node>& layer, const Precision &precision) const = 0;
    virtual bool canCreate(const std::shared_ptr<ngraph::Node>& node) const = 0;

    template <class T>
    static std::string asString(const T& value) {
        return std::to_string(value);
    }
};

template <class NGT>
class NodeConverter: public INodeConverter {
public:
    NodeConverter() = default;

    CNNLayer::Ptr createLayer(const std::shared_ptr<ngraph::Node>& layer, const Precision &precision) const override;

    bool canCreate(const std::shared_ptr<ngraph::Node>& node) const override {
        auto castedPtr = std::dynamic_pointer_cast<NGT>(node);
        return castedPtr != nullptr;
    }
};

class BaseConverter {
public:
    explicit BaseConverter(const std::string& type): type(type) {}
    virtual ~BaseConverter() = default;

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

        for (const auto& port : layer->getInputPorts()) {
            if (port.getParameters().find("type") == port.getParameters().end() ||
                    port.getData()->getData()->cbuffer() == nullptr)
                continue;
            res->blobs[port.getParameters().at("type")] = port.getData()->getData();
            if (weightLayerPtr == nullptr)
                continue;
            if (port.getParameters().at("type").as<std::string>() == "weights") {
                weightLayerPtr->_weights = port.getData()->getData();
            } else if (port.getParameters().at("type").as<std::string>() == "biases") {
                weightLayerPtr->_biases = port.getData()->getData();
            }
        }

        // For constant layers
        for (auto& it : layer->getParameters()) {
            if (it.second.is<Blob::CPtr>()) {
                res->blobs[it.first] = std::const_pointer_cast<Blob>(it.second.as<Blob::CPtr>());
            } else if (it.second.is<Blob::Ptr>()) {
                res->blobs[it.first] = it.second.as<Blob::Ptr>();
            }
        }

        res->params = convertParameters2Strings(layer->getParameters());
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

        auto typeIt = layer->getParameters().find("type");
        if (typeIt == layer->getParameters().end())
            THROW_IE_EXCEPTION << "Unsupported Activation layer. Type is unknown.";

        auto activationBuilder = activationCreators.find(typeIt->second);
        if (activationBuilder == activationCreators.end()) {
            THROW_IE_EXCEPTION << "Unsupported Activation layer type: " << typeIt->second.as<std::string>();
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

class RNNSequenceConverter: public BaseConverter {
public:
    RNNSequenceConverter(): BaseConverter("RNN") {}

    CNNLayer::Ptr createLayer(const std::shared_ptr<const ILayer>& layer, Precision precision) override {
        auto rnnLayer = LayerConverter<InferenceEngine::RNNSequenceLayer>("RNN").createLayer(layer, precision);
        rnnLayer->type = "RNN";
        std::string type = layer->getType();
        size_t pos = type.find("Sequence");
        if (pos != std::string::npos)
            type.erase(pos);
        rnnLayer->params["cell_type"] = type;
        return rnnLayer;
    }

    bool canCreate(const std::string& nodeType) const override {
        static const details::caseless_set<std::string> supportedRnnTypes {
            "LSTMSequence", "GRUSequence", "RNNSequence"
        };
        return supportedRnnTypes.find(nodeType) != supportedRnnTypes.end();
    }
};

}  // namespace Builder
}  // namespace InferenceEngine
