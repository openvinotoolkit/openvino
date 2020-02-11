// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_cnn_layer_builder.h>

#include <builders/ie_split_layer.hpp>
#include <string>
#include <vector>

using namespace InferenceEngine;

Builder::SplitLayer::SplitLayer(const std::string& name): LayerDecorator("Split", name) {
    getLayer()->getInputPorts().resize(1);
    setAxis(1);
}

Builder::SplitLayer::SplitLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("Split");
}

Builder::SplitLayer::SplitLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("Split");
}

Builder::SplitLayer& Builder::SplitLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::SplitLayer::getInputPort() const {
    return getLayer()->getInputPorts()[0];
}

Builder::SplitLayer& Builder::SplitLayer::setInputPort(const Port& port) {
    getLayer()->getInputPorts()[0] = port;
    return *this;
}

const std::vector<Port>& Builder::SplitLayer::getOutputPorts() const {
    return getLayer()->getOutputPorts();
}

Builder::SplitLayer& Builder::SplitLayer::setOutputPorts(const std::vector<Port>& ports) {
    getLayer()->getOutputPorts() = ports;
    return *this;
}

size_t Builder::SplitLayer::getAxis() const {
    return getLayer()->getParameters().at("axis");
}

Builder::SplitLayer& Builder::SplitLayer::setAxis(size_t axis) {
    getLayer()->getParameters()["axis"] = axis;
    return *this;
}

REG_CONVERTER_FOR(Split, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    layer.getParameters()["axis"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("axis", 1));
});