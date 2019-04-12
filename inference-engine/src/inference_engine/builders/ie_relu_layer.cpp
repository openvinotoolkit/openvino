// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_relu_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <string>

using namespace InferenceEngine;

Builder::ReLULayer::ReLULayer(const std::string& name): LayerDecorator("ReLU", name) {
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getInputPorts().resize(1);
    setNegativeSlope(0);
}

Builder::ReLULayer::ReLULayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("ReLU");
}

Builder::ReLULayer::ReLULayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("ReLU");
}

Builder::ReLULayer& Builder::ReLULayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::ReLULayer::getPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::ReLULayer& Builder::ReLULayer::setPort(const Port &port) {
    getLayer()->getOutputPorts()[0] = port;
    getLayer()->getInputPorts()[0] = port;
    return *this;
}

float Builder::ReLULayer::getNegativeSlope() const {
    return getLayer()->getParameters().at("negative_slope");
}

Builder::ReLULayer& Builder::ReLULayer::setNegativeSlope(float negativeSlope) {
    getLayer()->getParameters()["negative_slope"] = negativeSlope;
    return *this;
}

REG_VALIDATOR_FOR(ReLU, [] (const InferenceEngine::Builder::Layer::CPtr& input_layer, bool partial) {
    Builder::ReLULayer layer(input_layer);
    if (!input_layer->getInputPorts().empty() &&
        !input_layer->getOutputPorts().empty() &&
        !input_layer->getInputPorts()[0].shape().empty() &&
        !input_layer->getOutputPorts()[0].shape().empty() &&
        input_layer->getInputPorts()[0].shape() != input_layer->getOutputPorts()[0].shape()) {
        THROW_IE_EXCEPTION << "Input and output ports should be equal";
    }
});

REG_CONVERTER_FOR(ReLU, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    layer.getParameters()["negative_slope"] = cnnLayer->GetParamAsFloat("negative_slope", 0);
});
