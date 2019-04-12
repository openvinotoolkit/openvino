// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_grn_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <string>

using namespace InferenceEngine;

Builder::GRNLayer::GRNLayer(const std::string& name): LayerDecorator("GRN", name) {
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getInputPorts().resize(1);
    setBeta(0);
}

Builder::GRNLayer::GRNLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("GRN");
}

Builder::GRNLayer::GRNLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("GRN");
}

Builder::GRNLayer& Builder::GRNLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::GRNLayer::getPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::GRNLayer& Builder::GRNLayer::setPort(const Port &port) {
    getLayer()->getOutputPorts()[0] = port;
    getLayer()->getInputPorts()[0] = port;
    return *this;
}

float Builder::GRNLayer::getBeta() const {
    return getLayer()->getParameters().at("beta");
}

Builder::GRNLayer& Builder::GRNLayer::setBeta(float beta) {
    getLayer()->getParameters()["beta"] = beta;
    return *this;
}

REG_CONVERTER_FOR(GRN, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    layer.getParameters()["beta"] = static_cast<size_t>(cnnLayer->GetParamAsFloat("beta"));
});