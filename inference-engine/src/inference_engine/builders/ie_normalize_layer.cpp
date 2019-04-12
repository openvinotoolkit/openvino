// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_normalize_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <string>

using namespace InferenceEngine;

Builder::NormalizeLayer::NormalizeLayer(const std::string& name): LayerDecorator("Normalize", name) {
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getInputPorts().resize(1);
    setAcrossMaps(false);
    setChannelShared(false);
    setEpsilon(0.0000001f);
}

Builder::NormalizeLayer::NormalizeLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("Normalize");
}

Builder::NormalizeLayer::NormalizeLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("Normalize");
}

Builder::NormalizeLayer& Builder::NormalizeLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::NormalizeLayer::getPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::NormalizeLayer& Builder::NormalizeLayer::setPort(const Port &port) {
    getLayer()->getOutputPorts()[0] = port;
    getLayer()->getInputPorts()[0] = port;
    return *this;
}

bool Builder::NormalizeLayer::getAcrossMaps() const {
    return getLayer()->getParameters().at("region");
}

Builder::NormalizeLayer& Builder::NormalizeLayer::setAcrossMaps(bool acrossMap)  {
    getLayer()->getParameters()["region"] = acrossMap ? 1 : 0;
    return *this;
}

bool Builder::NormalizeLayer::getChannelShared() const {
    return getLayer()->getParameters().at("channel_shared");
}

Builder::NormalizeLayer& Builder::NormalizeLayer::setChannelShared(bool channelShared)  {
    getLayer()->getParameters()["channel_shared"] = channelShared ? 1 : 0;
    return *this;
}

float Builder::NormalizeLayer::getEpsilon() const {
    return getLayer()->getParameters().at("eps");
}

Builder::NormalizeLayer& Builder::NormalizeLayer::setEpsilon(float eps) {
    getLayer()->getParameters()["eps"] = eps;
    return *this;
}

REG_VALIDATOR_FOR(Normalize, [] (const InferenceEngine::Builder::Layer::CPtr& input_layer, bool partial) {
    Builder::NormalizeLayer layer(input_layer);
    if (layer.getEpsilon() <= 0) {
        THROW_IE_EXCEPTION << "Epsilon should be > 0";
    }
    if (!input_layer->getInputPorts().empty() &&
        !input_layer->getOutputPorts().empty() &&
        !input_layer->getInputPorts()[0].shape().empty() &&
        !input_layer->getOutputPorts()[0].shape().empty() &&
        input_layer->getInputPorts()[0].shape() != input_layer->getOutputPorts()[0].shape()) {
        THROW_IE_EXCEPTION << "Input and output ports should be equal";
    }
});

REG_CONVERTER_FOR(Normalize, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    layer.getParameters()["region"] = cnnLayer->GetParamsAsBool("region", 0);
    layer.getParameters()["channel_shared"] = cnnLayer->GetParamsAsBool("channel_shared", 0);
    layer.getParameters()["eps"] = cnnLayer->GetParamAsFloat("eps", 0);
});

