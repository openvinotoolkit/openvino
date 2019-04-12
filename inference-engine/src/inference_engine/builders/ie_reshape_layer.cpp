// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_reshape_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::ReshapeLayer::ReshapeLayer(const std::string& name): LayerDecorator("Reshape", name) {
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getInputPorts().resize(1);
}

Builder::ReshapeLayer::ReshapeLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("Reshape");
}

Builder::ReshapeLayer::ReshapeLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("Reshape");
}

Builder::ReshapeLayer& Builder::ReshapeLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::ReshapeLayer::getInputPort() const {
    return getLayer()->getInputPorts()[0];
}

Builder::ReshapeLayer& Builder::ReshapeLayer::setInputPort(const Port &port) {
    getLayer()->getInputPorts()[0] = port;
    return *this;
}

const Port& Builder::ReshapeLayer::getOutputPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::ReshapeLayer& Builder::ReshapeLayer::setOutputPort(const Port &port) {
    getLayer()->getOutputPorts()[0] = port;
    return *this;
}

const std::vector<int> Builder::ReshapeLayer::getDims() const {
    return getLayer()->getParameters().at("dim");
}

Builder::ReshapeLayer& Builder::ReshapeLayer::setDims(const std::vector<int>& dims) {
    getLayer()->getParameters()["dim"] = dims;
    return *this;
}

REG_CONVERTER_FOR(Flatten, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    layer.getParameters()["axis"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("axis", 0));
    layer.getParameters()["dim"] = cnnLayer->GetParamAsInts("dim", {});
});
REG_CONVERTER_FOR(Reshape, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    layer.getParameters()["axis"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("axis", 0));
    layer.getParameters()["dim"] = cnnLayer->GetParamAsInts("dim", {});
});
