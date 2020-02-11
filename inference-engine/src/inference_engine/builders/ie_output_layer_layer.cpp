// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_output_layer.hpp>

#include <string>

using namespace InferenceEngine;

Builder::OutputLayer::OutputLayer(const std::string& name): LayerDecorator("Output", name) {
    getLayer()->getInputPorts().resize(1);
}

Builder::OutputLayer::OutputLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("Output");
}

Builder::OutputLayer::OutputLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("Output");
}

Builder::OutputLayer& Builder::OutputLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::OutputLayer::getPort() const {
    return getLayer()->getInputPorts()[0];
}

Builder::OutputLayer& Builder::OutputLayer::setPort(const Port &port) {
    getLayer()->getInputPorts()[0] = port;
    return *this;
}

REG_VALIDATOR_FOR(Output, [] (const InferenceEngine::Builder::Layer::CPtr& input_layer, bool partial) {});
