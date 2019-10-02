// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_eltwise_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::EltwiseLayer::EltwiseLayer(const std::string& name): LayerDecorator("Eltwise", name) {
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getInputPorts().resize(2);
    setEltwiseType(EltwiseType::SUM);
}

Builder::EltwiseLayer::EltwiseLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("Eltwise");

    std::string operatorStr = getLayer()->getParameters()["operation"];
    if (operatorStr == "max") {
        type = MAX;
    } else if (operatorStr == "sum") {
        type = SUM;
    } else if (operatorStr == "mul") {
        type = MUL;
    } else if (operatorStr == "sub") {
        type = SUB;
    } else if (operatorStr == "div") {
        type = DIV;
    } else if (operatorStr == "min") {
        type = MIN;
    } else if (operatorStr == "squared_diff") {
        type = SQUARED_DIFF;
    }
}

Builder::EltwiseLayer::EltwiseLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("Eltwise");

    const auto cLayer = static_cast<const EltwiseLayer*>(this)->getLayer();

    std::string operatorStr = cLayer->getParameters().at("operation");
    if (operatorStr == "max") {
        type = MAX;
    } else if (operatorStr == "sum") {
        type = SUM;
    } else if (operatorStr == "mul") {
        type = MUL;
    } else if (operatorStr == "sub") {
        type = SUB;
    } else if (operatorStr == "div") {
        type = DIV;
    } else if (operatorStr == "min") {
        type = MIN;
    } else if (operatorStr == "squared_diff") {
        type = SQUARED_DIFF;
    }
}

Builder::EltwiseLayer& Builder::EltwiseLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const std::vector<Port>& Builder::EltwiseLayer::getInputPorts() const {
    return getLayer()->getInputPorts();
}

Builder::EltwiseLayer& Builder::EltwiseLayer::setInputPorts(const std::vector<Port>& ports) {
    getLayer()->getInputPorts() = ports;
    return *this;
}

const Port& Builder::EltwiseLayer::getOutputPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::EltwiseLayer& Builder::EltwiseLayer::setOutputPort(const Port &port) {
    getLayer()->getOutputPorts()[0] = port;
    return *this;
}

const std::vector<float> Builder::EltwiseLayer::getScales() const {
    return getLayer()->getParameters().at("scales");
}

// TODO: IR doesn't contain Scales!!!
Builder::EltwiseLayer& Builder::EltwiseLayer::setScales(const std::vector<float>& scales) {
    getLayer()->getParameters()["scales"] = scales;
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
            break;
        case SUB:
            operatorStr = "sub";
            break;
        case DIV:
            operatorStr = "div";
            break;
        case MIN:
            operatorStr = "min";
            break;
        case SQUARED_DIFF:
            operatorStr = "squared_diff";
            break;
    }
    getLayer()->getParameters()["operation"] = operatorStr;
    return *this;
}

REG_VALIDATOR_FOR(Eltwise, [](const InferenceEngine::Builder::Layer::CPtr& input_layer, bool partial) {
    Builder::EltwiseLayer layer(input_layer);

    if (layer.getInputPorts().size() < 2) {
        THROW_IE_EXCEPTION << "Input ports are incorrect in the layer " << layer.getName()
                           << ". Number of input ports should be >= 2.";
    }
    if (partial && (layer.getInputPorts()[0].shape().empty() || layer.getInputPorts()[1].shape().empty() ||
            layer.getOutputPort().shape().empty()))
        return;

    if (layer.getInputPorts()[0].shape() != layer.getInputPorts()[1].shape()) {
        THROW_IE_EXCEPTION << "Input ports are incorrect in the layer " << layer.getName()
                           << ". They should have equal dimensions";
    }

    if (layer.getInputPorts()[0].shape() != layer.getOutputPort().shape()) {
        THROW_IE_EXCEPTION << "Layer " << layer.getName() << " have different input and output ports. "
                           << "They should have equal dimensions.";
    }
});

REG_CONVERTER_FOR(Eltwise, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    layer.getParameters()["scales"] = cnnLayer->GetParamAsFloats("scales", {});
    layer.getParameters()["operation"] = cnnLayer->GetParamAsString("operation");
});
