// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_neghalf_log_layer.hpp>

#include <string>

using namespace InferenceEngine;

Builder::NegHalfLogLayer::NegHalfLogLayer(const std::string& name): LayerDecorator("NegHalfLog", name) {
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getInputPorts().resize(1);
}

Builder::NegHalfLogLayer::NegHalfLogLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("NegHalfLog");
}

Builder::NegHalfLogLayer::NegHalfLogLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("NegHalfLog");
}

Builder::NegHalfLogLayer& Builder::NegHalfLogLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::NegHalfLogLayer::getPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::NegHalfLogLayer& Builder::NegHalfLogLayer::setPort(const Port &port) {
    getLayer()->getOutputPorts()[0] = port;
    getLayer()->getInputPorts()[0] = port;
    return *this;
}
