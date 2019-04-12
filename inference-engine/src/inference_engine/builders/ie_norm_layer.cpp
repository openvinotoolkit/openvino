// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_norm_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <string>

using namespace InferenceEngine;

Builder::NormLayer::NormLayer(const std::string& name): LayerDecorator("Norm", name) {
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getInputPorts().resize(1);
    setAcrossMaps(false);
    setSize(0);
    setAlpha(0);
    setBeta(0);
}

Builder::NormLayer::NormLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("Norm");
}

Builder::NormLayer::NormLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("Norm");
}

Builder::NormLayer& Builder::NormLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::NormLayer::getPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::NormLayer& Builder::NormLayer::setPort(const Port &port) {
    getLayer()->getOutputPorts()[0] = port;
    getLayer()->getInputPorts()[0] = port;
    return *this;
}

size_t Builder::NormLayer::getSize() const {
    return getLayer()->getParameters().at("local-size");
}

Builder::NormLayer& Builder::NormLayer::setSize(size_t size) {
    getLayer()->getParameters()["local-size"] = size;
    return *this;
}

float Builder::NormLayer::getAlpha() const {
    return getLayer()->getParameters().at("alpha");
}

Builder::NormLayer& Builder::NormLayer::setAlpha(float alpha) {
    getLayer()->getParameters()["alpha"] = alpha;
    return *this;
}

float Builder::NormLayer::getBeta() const {
    return getLayer()->getParameters().at("beta");
}

Builder::NormLayer& Builder::NormLayer::setBeta(float beta) {
    getLayer()->getParameters()["beta"] = beta;
    return *this;
}

bool Builder::NormLayer::getAcrossMaps() const {
    return getLayer()->getParameters().at("region").as<std::string>() == "across";
}

Builder::NormLayer& Builder::NormLayer::setAcrossMaps(bool acrossMap)  {
    std::string value = acrossMap ? "across" : "same";
    getLayer()->getParameters()["region"] = value;
    return *this;
}

Builder::NormLayer::NormType Builder::NormLayer::getRegion() const {
    return getAcrossMaps() ? Builder::NormLayer::NormType::ACROSS_CHANNELS :
                             Builder::NormLayer::NormType::WITHIN_CHANNEL;
}
Builder::NormLayer& Builder::NormLayer::setRegion(Builder::NormLayer::NormType type) {
    setAcrossMaps(type);
    return *this;
}

REG_VALIDATOR_FOR(Norm, [] (const InferenceEngine::Builder::Layer::CPtr& input_layer, bool partial) {
    Builder::NormLayer layer(input_layer);
    if (layer.getAlpha() <= 0) {
        THROW_IE_EXCEPTION << "Alpha should be > 0";
    }
    if (layer.getBeta() <= 0) {
        THROW_IE_EXCEPTION << "Beta should be > 0";
    }
    if (layer.getSize() == 0) {
        THROW_IE_EXCEPTION << "Size should be > 0";
    }
    if (!input_layer->getInputPorts().empty() &&
        !input_layer->getOutputPorts().empty() &&
        !input_layer->getInputPorts()[0].shape().empty() &&
        !input_layer->getOutputPorts()[0].shape().empty() &&
        input_layer->getInputPorts()[0].shape() != input_layer->getOutputPorts()[0].shape()) {
        THROW_IE_EXCEPTION << "Input and output ports should be equal";
    }
});

REG_CONVERTER_FOR(Norm, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    layer.getParameters()["local-size"] = (size_t)cnnLayer->GetParamAsUInt("local-size", 0);
    layer.getParameters()["alpha"] = cnnLayer->GetParamAsFloat("alpha", 0);
    layer.getParameters()["beta"] = cnnLayer->GetParamAsFloat("beta", 0);
});
