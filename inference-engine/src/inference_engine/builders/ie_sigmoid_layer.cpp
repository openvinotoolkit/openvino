// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_sigmoid_layer.hpp>
#include <details/caseless.hpp>

#include <string>

using namespace InferenceEngine;

Builder::SigmoidLayer::SigmoidLayer(const std::string& name): LayerFragment("Sigmoid", name) {
    getLayer().getOutputPorts().resize(1);
    getLayer().getInputPorts().resize(1);
}

Builder::SigmoidLayer::SigmoidLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "Sigmoid"))
        THROW_IE_EXCEPTION << "Cannot create SigmoidLayer decorator for layer " << getLayer().getType();
}

Builder::SigmoidLayer& Builder::SigmoidLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}

const Port& Builder::SigmoidLayer::getPort() const {
    return getLayer().getOutputPorts()[0];
}

Builder::SigmoidLayer& Builder::SigmoidLayer::setPort(const Port &port) {
    getLayer().getOutputPorts()[0] = port;
    getLayer().getInputPorts()[0] = port;
    return *this;
}
