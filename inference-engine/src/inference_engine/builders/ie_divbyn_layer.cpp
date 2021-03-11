// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_divbyn_layer.hpp>

#include <string>

using namespace InferenceEngine;

Builder::DivByNLayer::DivByNLayer(const std::string& name): LayerDecorator("DivByN", name) {
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getInputPorts().resize(1);
}

Builder::DivByNLayer::DivByNLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("DivByN");
}

Builder::DivByNLayer::DivByNLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("DivByN");
}

Builder::DivByNLayer& Builder::DivByNLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::DivByNLayer::getPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::DivByNLayer& Builder::DivByNLayer::setPort(const Port &port) {
    getLayer()->getOutputPorts()[0] = port;
    getLayer()->getInputPorts()[0] = port;
    return *this;
}
