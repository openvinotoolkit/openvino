// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_softmax_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <string>

using namespace InferenceEngine;

Builder::SoftMaxLayer::SoftMaxLayer(const std::string& name): LayerDecorator("SoftMax", name) {
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getInputPorts().resize(1);
    setAxis(1);
}

Builder::SoftMaxLayer::SoftMaxLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("SoftMax");
}

Builder::SoftMaxLayer::SoftMaxLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("SoftMax");
}

Builder::SoftMaxLayer& Builder::SoftMaxLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::SoftMaxLayer::getPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::SoftMaxLayer& Builder::SoftMaxLayer::setPort(const Port &port) {
    getLayer()->getOutputPorts()[0] = port;
    getLayer()->getInputPorts()[0] = port;
    return *this;
}

size_t Builder::SoftMaxLayer::getAxis() const {
    return getLayer()->getParameters().at("axis");
}

Builder::SoftMaxLayer& Builder::SoftMaxLayer::setAxis(size_t axis) {
    getLayer()->getParameters()["axis"] = axis;
    return *this;
}

REG_CONVERTER_FOR(SoftMax, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    layer.getParameters()["axis"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("axis", 1));
});