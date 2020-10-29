// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_identity_layer.hpp>

#include <string>

using namespace InferenceEngine;

Builder::IdentityLayer::IdentityLayer(const std::string& name): LayerDecorator("Identity", name) {
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getInputPorts().resize(1);
}

Builder::IdentityLayer::IdentityLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("Identity");
}

Builder::IdentityLayer::IdentityLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("Identity");
}

Builder::IdentityLayer& Builder::IdentityLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::IdentityLayer::getPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::IdentityLayer& Builder::IdentityLayer::setPort(const Port &port) {
    getLayer()->getOutputPorts()[0] = port;
    getLayer()->getInputPorts()[0] = port;
    return *this;
}
