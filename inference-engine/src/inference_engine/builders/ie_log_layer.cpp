// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_log_layer.hpp>

#include <string>

using namespace InferenceEngine;

Builder::LogLayer::LogLayer(const std::string& name): LayerDecorator("Log", name) {
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getInputPorts().resize(1);
}

Builder::LogLayer::LogLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("Log");
}

Builder::LogLayer::LogLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("Log");
}

Builder::LogLayer& Builder::LogLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::LogLayer::getPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::LogLayer& Builder::LogLayer::setPort(const Port &port) {
    getLayer()->getOutputPorts()[0] = port;
    getLayer()->getInputPorts()[0] = port;
    return *this;
}
