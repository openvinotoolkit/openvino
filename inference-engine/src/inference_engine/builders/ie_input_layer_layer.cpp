// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_input_layer.hpp>
#include <details/caseless.hpp>

#include <string>

using namespace InferenceEngine;

Builder::InputLayer::InputLayer(const std::string& name): LayerFragment("Input", name) {
    getLayer().getOutputPorts().resize(1);
}

Builder::InputLayer::InputLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "Input"))
        THROW_IE_EXCEPTION << "Cannot create InputLayer decorator for layer " << getLayer().getType();
}

Builder::InputLayer& Builder::InputLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}

const Port& Builder::InputLayer::getPort() const {
    return getLayer().getOutputPorts()[0];
}

Builder::InputLayer& Builder::InputLayer::setPort(const Port &port) {
    getLayer().getOutputPorts()[0] = port;
    return *this;
}

void Builder::InputLayer::validate(const Layer& layer) {
    if (layer.getOutputPorts()[0].shape().empty())
        THROW_IE_EXCEPTION << layer.getType() << " node " << layer.getName() << " should have shape!";
}

REG_VALIDATOR_FOR(Input,  Builder::InputLayer::validate);