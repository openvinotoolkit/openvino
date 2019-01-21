// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_relu_layer.hpp>
#include <details/caseless.hpp>

#include <string>

using namespace InferenceEngine;

Builder::ReLULayer::ReLULayer(const std::string& name): LayerFragment("ReLU", name) {
    getLayer().getOutputPorts().resize(1);
    getLayer().getInputPorts().resize(1);
    setNegativeSlope(0);
}

Builder::ReLULayer::ReLULayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "ReLU"))
        THROW_IE_EXCEPTION << "Cannot create ReLULayer decorator for layer " << getLayer().getType();
}

Builder::ReLULayer& Builder::ReLULayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}

const Port& Builder::ReLULayer::getPort() const {
    return getLayer().getOutputPorts()[0];
}

Builder::ReLULayer& Builder::ReLULayer::setPort(const Port &port) {
    getLayer().getOutputPorts()[0] = port;
    getLayer().getInputPorts()[0] = port;
    return *this;
}

float Builder::ReLULayer::getNegativeSlope() const {
    return getLayer().getParameters()["negative_slope"].asFloat();
}

Builder::ReLULayer& Builder::ReLULayer::setNegativeSlope(float negativeSlope) {
    getLayer().getParameters()["negative_slope"] = negativeSlope;
    return *this;
}
