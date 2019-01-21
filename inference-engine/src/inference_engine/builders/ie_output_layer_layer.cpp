// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_output_layer.hpp>
#include <details/caseless.hpp>

#include <string>

using namespace InferenceEngine;

Builder::OutputLayer::OutputLayer(const std::string& name): LayerFragment("Output", name) {
    getLayer().getInputPorts().resize(1);
}

Builder::OutputLayer::OutputLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "Output"))
        THROW_IE_EXCEPTION << "Cannot create OutputLayer decorator for layer " << getLayer().getType();
}

Builder::OutputLayer& Builder::OutputLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}

const Port& Builder::OutputLayer::getPort() const {
    return getLayer().getInputPorts()[0];
}

Builder::OutputLayer& Builder::OutputLayer::setPort(const Port &port) {
    getLayer().getInputPorts()[0] = port;
    return *this;
}
