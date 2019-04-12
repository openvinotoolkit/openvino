// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_tanh_layer.hpp>

#include <string>

using namespace InferenceEngine;

Builder::TanHLayer::TanHLayer(const std::string& name): LayerDecorator("TanH", name) {
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getInputPorts().resize(1);
}

Builder::TanHLayer::TanHLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("TanH");
}

Builder::TanHLayer::TanHLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("TanH");
}

Builder::TanHLayer& Builder::TanHLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::TanHLayer::getPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::TanHLayer& Builder::TanHLayer::setPort(const Port &port) {
    getLayer()->getOutputPorts()[0] = port;
    getLayer()->getInputPorts()[0] = port;
    return *this;
}

REG_VALIDATOR_FOR(TanH, [] (const InferenceEngine::Builder::Layer::CPtr& input_layer, bool partial) {
    if (!input_layer->getInputPorts().empty() &&
        !input_layer->getOutputPorts().empty() &&
        !input_layer->getInputPorts()[0].shape().empty() &&
        !input_layer->getOutputPorts()[0].shape().empty() &&
        input_layer->getInputPorts()[0].shape() != input_layer->getOutputPorts()[0].shape()) {
        THROW_IE_EXCEPTION << "Input and output ports should be equal";
    }
});
