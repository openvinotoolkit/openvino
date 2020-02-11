// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_relu6_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <string>

using namespace InferenceEngine;

Builder::ReLU6Layer::ReLU6Layer(const std::string& name): LayerDecorator("ReLU6", name) {
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getInputPorts().resize(1);
    setN(6);
}

Builder::ReLU6Layer::ReLU6Layer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("ReLU6");
}

Builder::ReLU6Layer::ReLU6Layer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("ReLU6");
}

Builder::ReLU6Layer& Builder::ReLU6Layer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::ReLU6Layer::getPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::ReLU6Layer& Builder::ReLU6Layer::setPort(const Port &port) {
    getLayer()->getOutputPorts()[0] = port;
    getLayer()->getInputPorts()[0] = port;
    return *this;
}

float Builder::ReLU6Layer::getN() const {
    return getLayer()->getParameters().at("n");
}

Builder::ReLU6Layer& Builder::ReLU6Layer::setN(float n) {
    getLayer()->getParameters()["n"] = n;
    return *this;
}

REG_VALIDATOR_FOR(ReLU6, [] (const InferenceEngine::Builder::Layer::CPtr& input_layer, bool partial) {
    if (!input_layer->getInputPorts().empty() &&
        !input_layer->getOutputPorts().empty() &&
        !input_layer->getInputPorts()[0].shape().empty() &&
        !input_layer->getOutputPorts()[0].shape().empty() &&
        input_layer->getInputPorts()[0].shape() != input_layer->getOutputPorts()[0].shape()) {
        THROW_IE_EXCEPTION << "Input and output ports should be equal";
    }
});

REG_CONVERTER_FOR(ReLU6, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    layer.getParameters()["n"] = cnnLayer->GetParamAsFloat("n", 0);
});
