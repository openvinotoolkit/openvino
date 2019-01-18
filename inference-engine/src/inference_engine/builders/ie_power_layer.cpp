// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_power_layer.hpp>
#include <details/caseless.hpp>

#include <string>

using namespace InferenceEngine;

Builder::PowerLayer::PowerLayer(const std::string& name): LayerFragment("Power", name) {
    getLayer().getOutputPorts().resize(1);
    getLayer().getInputPorts().resize(1);
    setPower(1);
    setScale(1);
    setShift(0);
}

Builder::PowerLayer::PowerLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "Power"))
        THROW_IE_EXCEPTION << "Cannot create PowerLayer decorator for layer " << getLayer().getType();
}

Builder::PowerLayer& Builder::PowerLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}

const Port& Builder::PowerLayer::getPort() const {
    return getLayer().getOutputPorts()[0];
}

Builder::PowerLayer& Builder::PowerLayer::setPort(const Port &port) {
    getLayer().getOutputPorts()[0] = port;
    getLayer().getInputPorts()[0] = port;
    return *this;
}

float Builder::PowerLayer::getPower() const {
    return getLayer().getParameters()["power"].asFloat();
}

Builder::PowerLayer& Builder::PowerLayer::setPower(float power) {
    getLayer().getParameters()["power"] = power;
    return *this;
}

float Builder::PowerLayer::getScale() const {
    return getLayer().getParameters()["scale"].asFloat();
}

Builder::PowerLayer& Builder::PowerLayer::setScale(float scale) {
    getLayer().getParameters()["scale"] = scale;
    return *this;
}

float Builder::PowerLayer::getShift() const {
    return getLayer().getParameters()["shift"].asFloat();
}

Builder::PowerLayer& Builder::PowerLayer::setShift(float shift) {
    getLayer().getParameters()["shift"] = shift;
    return *this;
}

