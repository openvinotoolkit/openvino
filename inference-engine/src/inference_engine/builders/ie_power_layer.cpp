// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_power_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <string>

using namespace InferenceEngine;

Builder::PowerLayer::PowerLayer(const std::string& name): LayerDecorator("Power", name) {
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getInputPorts().resize(1);
    setPower(1);
    setScale(1);
    setShift(0);
}

Builder::PowerLayer::PowerLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("Power");
}

Builder::PowerLayer::PowerLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("Power");
}

Builder::PowerLayer& Builder::PowerLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::PowerLayer::getPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::PowerLayer& Builder::PowerLayer::setPort(const Port &port) {
    getLayer()->getOutputPorts()[0] = port;
    getLayer()->getInputPorts()[0] = port;
    return *this;
}

float Builder::PowerLayer::getPower() const {
    return getLayer()->getParameters().at("power");
}

Builder::PowerLayer& Builder::PowerLayer::setPower(float power) {
    getLayer()->getParameters()["power"] = power;
    return *this;
}

float Builder::PowerLayer::getScale() const {
    return getLayer()->getParameters().at("scale");
}

Builder::PowerLayer& Builder::PowerLayer::setScale(float scale) {
    getLayer()->getParameters()["scale"] = scale;
    return *this;
}

float Builder::PowerLayer::getShift() const {
    return getLayer()->getParameters().at("shift");
}

Builder::PowerLayer& Builder::PowerLayer::setShift(float shift) {
    getLayer()->getParameters()["shift"] = shift;
    return *this;
}

REG_CONVERTER_FOR(Power, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    layer.getParameters()["shift"] = cnnLayer->GetParamAsFloat("shift", 0);
    layer.getParameters()["scale"] = cnnLayer->GetParamAsFloat("scale", 1);
    layer.getParameters()["power"] = cnnLayer->GetParamAsFloat("power", 1);
});
