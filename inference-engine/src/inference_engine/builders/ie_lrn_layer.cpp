// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_lrn_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <string>

using namespace InferenceEngine;

Builder::LRNLayer::LRNLayer(const std::string& name): LayerDecorator("LRN", name) {
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getInputPorts().resize(1);
    setSize(1);
    setAlpha(1e-4);
    setBeta(0.75f);
    setBias(1.0f);
}

Builder::LRNLayer::LRNLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("LRN");
}

Builder::LRNLayer::LRNLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("LRN");
}

Builder::LRNLayer& Builder::LRNLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::LRNLayer::getPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::LRNLayer& Builder::LRNLayer::setPort(const Port &port) {
    getLayer()->getOutputPorts()[0] = port;
    getLayer()->getInputPorts()[0] = port;
    return *this;
}

size_t Builder::LRNLayer::getSize() const {
    return getLayer()->getParameters().at("size");
}

Builder::LRNLayer& Builder::LRNLayer::setSize(size_t size) {
    getLayer()->getParameters()["size"] = size;
    return *this;
}

float Builder::LRNLayer::getAlpha() const {
    return getLayer()->getParameters().at("alpha");
}

Builder::LRNLayer& Builder::LRNLayer::setAlpha(float alpha) {
    getLayer()->getParameters()["alpha"] = alpha;
    return *this;
}

float Builder::LRNLayer::getBeta() const {
    return getLayer()->getParameters().at("beta");
}

Builder::LRNLayer& Builder::LRNLayer::setBeta(float beta) {
    getLayer()->getParameters()["beta"] = beta;
    return *this;
}

float Builder::LRNLayer::getBias() const {
    return getLayer()->getParameters().at("bias");
}

Builder::LRNLayer& Builder::LRNLayer::setBias(float bias) {
    getLayer()->getParameters()["bias"] = bias;
    return *this;
}

REG_VALIDATOR_FOR(LRN, [](const Builder::Layer::CPtr &input_layer, bool partial) {
    Builder::LRNLayer layer(input_layer);
    if (layer.getAlpha() <= 0) {
        THROW_IE_EXCEPTION << "Alpha should be > 0";
    }
    if (layer.getBeta() <= 0) {
        THROW_IE_EXCEPTION << "Beta should be > 0";
    }
    if (layer.getSize() == 0) {
        THROW_IE_EXCEPTION << "Size should be > 0";
    }
    if (!input_layer->getInputPorts().empty() &&
        !input_layer->getOutputPorts().empty() &&
        !input_layer->getInputPorts()[0].shape().empty() &&
        !input_layer->getOutputPorts()[0].shape().empty() &&
        input_layer->getInputPorts()[0].shape() != input_layer->getOutputPorts()[0].shape()) {
        THROW_IE_EXCEPTION << "Input and output ports should be equal";
    }
});

REG_CONVERTER_FOR(LRN, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    layer.getParameters()["bias"] = cnnLayer->GetParamAsFloat("bias", 1.0f);
    layer.getParameters()["beta"] = cnnLayer->GetParamAsFloat("beta", 0.75f);
    layer.getParameters()["alpha"] = cnnLayer->GetParamAsFloat("alpha", 1e-4f);
    layer.getParameters()["size"] = cnnLayer->GetParamAsUInt("size", 1);
});