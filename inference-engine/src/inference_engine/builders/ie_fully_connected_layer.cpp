// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_fully_connected_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::FullyConnectedLayer::FullyConnectedLayer(const std::string& name): LayerDecorator("FullyConnected", name) {
    getLayer()->getInputPorts().resize(3);
    getLayer()->getInputPorts()[1].setParameter("type", "weights");
    getLayer()->getInputPorts()[2].setParameter("type", "biases");
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getParameters()["out-size"] = 0;
}

Builder::FullyConnectedLayer::FullyConnectedLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("FullyConnected");
}

Builder::FullyConnectedLayer::FullyConnectedLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("FullyConnected");
}

Builder::FullyConnectedLayer &Builder::FullyConnectedLayer::setName(const std::string &name) {
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::FullyConnectedLayer::getInputPort() const {
    return getLayer()->getInputPorts()[0];
}

Builder::FullyConnectedLayer& Builder::FullyConnectedLayer::setInputPort(const Port& port) {
    getLayer()->getInputPorts()[0] = port;
    return *this;
}

const Port& Builder::FullyConnectedLayer::getOutputPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::FullyConnectedLayer& Builder::FullyConnectedLayer::setOutputPort(const Port& port) {
    getLayer()->getOutputPorts()[0] = port;
    return *this;
}

size_t Builder::FullyConnectedLayer::getOutputNum() const {
    return getLayer()->getParameters().at("out-size");
}

Builder::FullyConnectedLayer& Builder::FullyConnectedLayer::setOutputNum(size_t outNum) {
    getLayer()->getParameters()["out-size"] = outNum;
    return *this;
}

REG_VALIDATOR_FOR(FullyConnected, [](const InferenceEngine::Builder::Layer::CPtr& layer, bool partial) {
});

REG_CONVERTER_FOR(FullyConnected, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    layer.getParameters()["out-size"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("out-size", 0));
});
