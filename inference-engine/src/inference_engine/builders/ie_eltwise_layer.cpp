// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_eltwise_layer.hpp>
#include <details/caseless.hpp>

#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::EltwiseLayer::EltwiseLayer(const std::string& name): LayerFragment("Eltwise", name) {
    getLayer().getOutputPorts().resize(1);
    setEltwiseType(EltwiseType::SUM);
}

Builder::EltwiseLayer::EltwiseLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "Eltwise"))
        THROW_IE_EXCEPTION << "Cannot create EltwiseLayer decorator for layer " << getLayer().getType();

    std::string operatorStr = getLayer().getParameters()["operation"];
    if (operatorStr == "max") {
        type = MAX;
    } else if (operatorStr == "sum") {
        type = SUM;
    } else if (operatorStr == "mul") {
        type = MUL;
    }
}

Builder::EltwiseLayer& Builder::EltwiseLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}

const std::vector<Port>& Builder::EltwiseLayer::getInputPorts() const {
    return getLayer().getInputPorts();
}

Builder::EltwiseLayer& Builder::EltwiseLayer::setInputPorts(const std::vector<Port>& ports) {
    getLayer().getInputPorts() = ports;
    return *this;
}

const Port& Builder::EltwiseLayer::getOutputPort() const {
    return getLayer().getOutputPorts()[0];
}

Builder::EltwiseLayer& Builder::EltwiseLayer::setOutputPort(const Port &port) {
    getLayer().getOutputPorts()[0] = port;
    return *this;
}

const std::vector<float> Builder::EltwiseLayer::getScales() const {
    return getLayer().getParameters()["scales"].asFloats({});
}

// TODO: IR doesn't contain Scales!!!
Builder::EltwiseLayer& Builder::EltwiseLayer::setScales(const std::vector<float>& scales) {
    getLayer().getParameters()["scales"] = scales;
    return *this;
}

Builder::EltwiseLayer::EltwiseType Builder::EltwiseLayer::getEltwiseType() const {
    return type;
}

Builder::EltwiseLayer& Builder::EltwiseLayer::setEltwiseType(Builder::EltwiseLayer::EltwiseType type) {
    this->type = type;
    std::string operatorStr;
    switch (type) {
    case MAX:
        operatorStr = "max";
        break;
    case SUM:
        operatorStr = "sum";
        break;
    case MUL:
        operatorStr = "mul";
    }
    getLayer().getParameters()["operation"] = operatorStr;
    return *this;
}


