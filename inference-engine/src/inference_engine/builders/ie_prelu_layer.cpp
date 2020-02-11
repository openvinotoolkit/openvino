// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_prelu_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <string>

using namespace InferenceEngine;

Builder::PReLULayer::PReLULayer(const std::string& name): LayerDecorator("PReLU", name) {
    getLayer()->getInputPorts().resize(2);
    getLayer()->getInputPorts()[1].setParameter("type", "weights");
    getLayer()->getOutputPorts().resize(1);
    setChannelShared(false);
}

Builder::PReLULayer::PReLULayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("PReLU");
}

Builder::PReLULayer::PReLULayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("PReLU");
}

Builder::PReLULayer& Builder::PReLULayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::PReLULayer::getPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::PReLULayer& Builder::PReLULayer::setPort(const Port &port) {
    getLayer()->getOutputPorts()[0] = port;
    getLayer()->getInputPorts()[0] = port;
    return *this;
}

bool Builder::PReLULayer::getChannelShared() const {
    return getLayer()->getParameters().at("channel_shared");
}
Builder::PReLULayer& Builder::PReLULayer::setChannelShared(bool flag) {
    getLayer()->getParameters()["channel_shared"] = flag ? 1 : 0;
    return *this;
}

REG_CONVERTER_FOR(PReLU, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    layer.getParameters()["channel_shared"] = cnnLayer->GetParamAsBool("channel_shared", false);
});