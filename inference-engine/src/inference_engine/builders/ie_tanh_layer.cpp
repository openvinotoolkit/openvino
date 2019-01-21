// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_tanh_layer.hpp>
#include <details/caseless.hpp>

#include <string>

using namespace InferenceEngine;

Builder::TanHLayer::TanHLayer(const std::string& name): LayerFragment("TanH", name) {
    getLayer().getOutputPorts().resize(1);
    getLayer().getInputPorts().resize(1);
}

Builder::TanHLayer::TanHLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "TanH"))
        THROW_IE_EXCEPTION << "Cannot create TanHLayer decorator for layer " << getLayer().getType();
}

Builder::TanHLayer& Builder::TanHLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}

const Port& Builder::TanHLayer::getPort() const {
    return getLayer().getOutputPorts()[0];
}

Builder::TanHLayer& Builder::TanHLayer::setPort(const Port &port) {
    getLayer().getOutputPorts()[0] = port;
    getLayer().getInputPorts()[0] = port;
    return *this;
}