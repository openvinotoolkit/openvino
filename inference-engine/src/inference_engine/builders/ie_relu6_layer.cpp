// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_relu6_layer.hpp>
#include <details/caseless.hpp>

#include <string>

using namespace InferenceEngine;

Builder::ReLU6Layer::ReLU6Layer(const std::string& name): LayerFragment("ReLU6", name) {
    getLayer().getOutputPorts().resize(1);
    getLayer().getInputPorts().resize(1);
    setN(6);
}

Builder::ReLU6Layer::ReLU6Layer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "ReLU6"))
        THROW_IE_EXCEPTION << "Cannot create ReLU6Layer decorator for layer " << getLayer().getType();
}

Builder::ReLU6Layer& Builder::ReLU6Layer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}

const Port& Builder::ReLU6Layer::getPort() const {
    return getLayer().getOutputPorts()[0];
}

Builder::ReLU6Layer& Builder::ReLU6Layer::setPort(const Port &port) {
    getLayer().getOutputPorts()[0] = port;
    getLayer().getInputPorts()[0] = port;
    return *this;
}

float Builder::ReLU6Layer::getN() const {
    return getLayer().getParameters()["n"].asFloat();
}

Builder::ReLU6Layer& Builder::ReLU6Layer::setN(float n) {
    getLayer().getParameters()["n"] = n;
    return *this;
}


