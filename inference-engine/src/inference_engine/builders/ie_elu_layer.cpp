// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_elu_layer.hpp>
#include <details/caseless.hpp>

#include <string>

using namespace InferenceEngine;

Builder::ELULayer::ELULayer(const std::string& name): LayerFragment("ELU", name) {
    getLayer().getOutputPorts().resize(1);
    getLayer().getInputPorts().resize(1);
    setAlpha(1);
}

Builder::ELULayer::ELULayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "ELU"))
        THROW_IE_EXCEPTION << "Cannot create ELULayer decorator for layer " << getLayer().getType();
}

Builder::ELULayer& Builder::ELULayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}

const Port& Builder::ELULayer::getPort() const {
    return getLayer().getOutputPorts()[0];
}

Builder::ELULayer& Builder::ELULayer::setPort(const Port &port) {
    getLayer().getOutputPorts()[0] = port;
    getLayer().getInputPorts()[0] = port;
    return *this;
}

float Builder::ELULayer::getAlpha() const {
    return getLayer().getParameters()["alpha"].asFloat();
}

Builder::ELULayer& Builder::ELULayer::setAlpha(float alpha) {
    getLayer().getParameters()["alpha"] = alpha;
    return *this;
}

