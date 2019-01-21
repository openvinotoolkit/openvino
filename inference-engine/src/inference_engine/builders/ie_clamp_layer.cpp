// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_clamp_layer.hpp>
#include <details/caseless.hpp>

#include <string>

using namespace InferenceEngine;

Builder::ClampLayer::ClampLayer(const std::string& name): LayerFragment("Clamp", name) {
    getLayer().getOutputPorts().resize(1);
    getLayer().getInputPorts().resize(1);
    setMinValue(0.0f);
    setMaxValue(1.0f);
}

Builder::ClampLayer::ClampLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "Clamp"))
        THROW_IE_EXCEPTION << "Cannot create ClampLayer decorator for layer " << getLayer().getType();
}

Builder::ClampLayer& Builder::ClampLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}

const Port& Builder::ClampLayer::getPort() const {
    return getLayer().getOutputPorts()[0];
}

Builder::ClampLayer& Builder::ClampLayer::setPort(const Port &port) {
    getLayer().getOutputPorts()[0] = port;
    getLayer().getInputPorts()[0] = port;
    return *this;
}

float Builder::ClampLayer::getMaxValue() const {
    return getLayer().getParameters()["max"].asFloat();
}

Builder::ClampLayer& Builder::ClampLayer::setMaxValue(float maxValue) {
    getLayer().getParameters()["max"] = maxValue;
    return *this;
}

float Builder::ClampLayer::getMinValue() const {
    return getLayer().getParameters()["min"].asFloat();
}

Builder::ClampLayer& Builder::ClampLayer::setMinValue(float minValue) {
    getLayer().getParameters()["min"] = minValue;
    return *this;
}

