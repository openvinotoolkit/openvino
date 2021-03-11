// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_exp_layer.hpp>

#include <string>

using namespace InferenceEngine;

Builder::ExpLayer::ExpLayer(const std::string& name): LayerDecorator("Exp", name) {
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getInputPorts().resize(1);
}

Builder::ExpLayer::ExpLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("Exp");
}

Builder::ExpLayer::ExpLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("Exp");
}

Builder::ExpLayer& Builder::ExpLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::ExpLayer::getPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::ExpLayer& Builder::ExpLayer::setPort(const Port &port) {
    getLayer()->getOutputPorts()[0] = port;
    getLayer()->getInputPorts()[0] = port;
    return *this;
}
