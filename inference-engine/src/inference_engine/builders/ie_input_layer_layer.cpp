// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_input_layer.hpp>
#include <string>

using namespace InferenceEngine;

Builder::InputLayer::InputLayer(const std::string& name): LayerDecorator("Input", name) {
    getLayer()->getOutputPorts().resize(1);
}

Builder::InputLayer::InputLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("Input");
}

Builder::InputLayer::InputLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("Input");
}

Builder::InputLayer& Builder::InputLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::InputLayer::getPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::InputLayer& Builder::InputLayer::setPort(const Port& port) {
    getLayer()->getOutputPorts()[0] = port;
    return *this;
}

REG_VALIDATOR_FOR(Input, [](const InferenceEngine::Builder::Layer::CPtr& layer, bool partial) {
    if (layer->getOutputPorts()[0].shape().empty())
        THROW_IE_EXCEPTION << layer->getType() << " node " << layer->getName() << " should have shape!";
});
