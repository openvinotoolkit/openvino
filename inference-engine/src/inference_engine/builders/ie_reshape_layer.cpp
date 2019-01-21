// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_reshape_layer.hpp>
#include <details/caseless.hpp>

#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::ReshapeLayer::ReshapeLayer(const std::string& name): LayerFragment("Reshape", name) {
    getLayer().getOutputPorts().resize(1);
    getLayer().getInputPorts().resize(1);
}

Builder::ReshapeLayer::ReshapeLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "Reshape"))
        THROW_IE_EXCEPTION << "Cannot create ReshapeLayer decorator for layer " << getLayer().getType();
}

Builder::ReshapeLayer& Builder::ReshapeLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}

const Port& Builder::ReshapeLayer::getInputPort() const {
    return getLayer().getInputPorts()[0];
}

Builder::ReshapeLayer& Builder::ReshapeLayer::setInputPort(const Port &port) {
    getLayer().getInputPorts()[0] = port;
    return *this;
}

const Port& Builder::ReshapeLayer::getOutputPort() const {
    return getLayer().getOutputPorts()[0];
}

Builder::ReshapeLayer& Builder::ReshapeLayer::setOutputPort(const Port &port) {
    getLayer().getOutputPorts()[0] = port;
    return *this;
}

const std::vector<int> Builder::ReshapeLayer::getDims() const {
    return getLayer().getParameters()["dim"].asInts();
}

Builder::ReshapeLayer& Builder::ReshapeLayer::setDims(const std::vector<int>& dims) {
    getLayer().getParameters()["dim"] = dims;
    return *this;
}

