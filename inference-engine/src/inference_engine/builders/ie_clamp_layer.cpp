// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_clamp_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <string>

using namespace InferenceEngine;

Builder::ClampLayer::ClampLayer(const std::string& name): LayerDecorator("Clamp", name) {
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getInputPorts().resize(1);
    setMinValue(0.0f);
    setMaxValue(1.0f);
}

Builder::ClampLayer::ClampLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("Clamp");
}

Builder::ClampLayer::ClampLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("Clamp");
}

Builder::ClampLayer& Builder::ClampLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::ClampLayer::getPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::ClampLayer& Builder::ClampLayer::setPort(const Port &port) {
    getLayer()->getOutputPorts()[0] = port;
    getLayer()->getInputPorts()[0] = port;
    return *this;
}

float Builder::ClampLayer::getMaxValue() const {
    return getLayer()->getParameters().at("max");
}

Builder::ClampLayer& Builder::ClampLayer::setMaxValue(float maxValue) {
    getLayer()->getParameters()["max"] = maxValue;
    return *this;
}

float Builder::ClampLayer::getMinValue() const {
    return getLayer()->getParameters().at("min");
}

Builder::ClampLayer& Builder::ClampLayer::setMinValue(float minValue) {
    getLayer()->getParameters()["min"] = minValue;
    return *this;
}

REG_VALIDATOR_FOR(Clamp, [](const InferenceEngine::Builder::Layer::CPtr& input_layer, bool partial) {
    Builder::ClampLayer layer(input_layer);
    if (layer.getMinValue() > layer.getMaxValue()) {
        THROW_IE_EXCEPTION << "MinValue should be less or equal MaxValue";
    }
    if (!input_layer->getInputPorts().empty() &&
        !input_layer->getOutputPorts().empty() &&
        !input_layer->getInputPorts()[0].shape().empty() &&
        !input_layer->getOutputPorts()[0].shape().empty() &&
        input_layer->getInputPorts()[0].shape() != input_layer->getOutputPorts()[0].shape()) {
        THROW_IE_EXCEPTION << "Input and output ports should be equal";
    }
});

REG_CONVERTER_FOR(Clamp, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    layer.getParameters()["max"] = cnnLayer->GetParamAsFloat("max", 0);
    layer.getParameters()["min"] = cnnLayer->GetParamAsFloat("min", 0);
});
