// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_elu_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <string>

using namespace InferenceEngine;

Builder::ELULayer::ELULayer(const std::string& name): LayerDecorator("ELU", name) {
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getInputPorts().resize(1);
    setAlpha(1);
}

Builder::ELULayer::ELULayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("ELU");
}

Builder::ELULayer::ELULayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("ELU");
}

Builder::ELULayer& Builder::ELULayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::ELULayer::getPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::ELULayer& Builder::ELULayer::setPort(const Port &port) {
    getLayer()->getOutputPorts()[0] = port;
    getLayer()->getInputPorts()[0] = port;
    return *this;
}

float Builder::ELULayer::getAlpha() const {
    return getLayer()->getParameters().at("alpha");
}

Builder::ELULayer& Builder::ELULayer::setAlpha(float alpha) {
    getLayer()->getParameters()["alpha"] = alpha;
    return *this;
}

REG_VALIDATOR_FOR(ELU, [] (const InferenceEngine::Builder::Layer::CPtr& input_layer, bool partial) {
    if (!input_layer->getInputPorts().empty() &&
        !input_layer->getOutputPorts().empty() &&
        !input_layer->getInputPorts()[0].shape().empty() &&
        !input_layer->getOutputPorts()[0].shape().empty() &&
        input_layer->getInputPorts()[0].shape() != input_layer->getOutputPorts()[0].shape()) {
        THROW_IE_EXCEPTION << "Input and output ports should be equal";
    }
    Builder::ELULayer layer(input_layer);
    if (layer.getAlpha() < 0) {
        THROW_IE_EXCEPTION << "Alpha should be >= 0";
    }
});

REG_CONVERTER_FOR(ELU, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    layer.getParameters()["alpha"] = cnnLayer->GetParamAsFloat("alpha", 0);
});

