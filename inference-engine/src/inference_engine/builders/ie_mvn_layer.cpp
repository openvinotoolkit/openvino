// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_mvn_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <string>

using namespace InferenceEngine;

Builder::MVNLayer::MVNLayer(const std::string& name): LayerDecorator("MVN", name) {
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getInputPorts().resize(1);
    setEpsilon(9.999999717180685e-10f);
    setNormalize(true);
    setAcrossChannels(true);
}

Builder::MVNLayer::MVNLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("MVN");
}

Builder::MVNLayer::MVNLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("MVN");
}

Builder::MVNLayer& Builder::MVNLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::MVNLayer::getPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::MVNLayer& Builder::MVNLayer::setPort(const Port &port) {
    getLayer()->getOutputPorts()[0] = port;
    getLayer()->getInputPorts()[0] = port;
    return *this;
}

bool Builder::MVNLayer::getAcrossChannels() const {
    return getLayer()->getParameters().at("across_channels");
}
Builder::MVNLayer& Builder::MVNLayer::setAcrossChannels(bool flag) {
    getLayer()->getParameters()["across_channels"] = flag ? 1 : 0;
    return *this;
}
bool Builder::MVNLayer::getNormalize() const {
    return getLayer()->getParameters().at("normalize_variance");
}
Builder::MVNLayer& Builder::MVNLayer::setNormalize(bool flag) {
    getLayer()->getParameters()["normalize_variance"] = flag ? 1 : 0;
    return *this;
}
float Builder::MVNLayer::getEpsilon() const {
    return getLayer()->getParameters().at("eps");
}
Builder::MVNLayer& Builder::MVNLayer::setEpsilon(float eps) {
    getLayer()->getParameters()["eps"] = eps;
    return *this;
}

REG_VALIDATOR_FOR(MVN, [](const Builder::Layer::CPtr& input_layer, bool partial) {
    Builder::MVNLayer layer(input_layer);
    if (layer.getEpsilon() <= 0) {
        THROW_IE_EXCEPTION << "Epsilon should be > 0";
    }
    if (!input_layer->getInputPorts().empty() &&
        !input_layer->getOutputPorts().empty() &&
        !input_layer->getInputPorts()[0].shape().empty() &&
        !input_layer->getOutputPorts()[0].shape().empty() &&
        input_layer->getInputPorts()[0].shape() != input_layer->getOutputPorts()[0].shape()) {
        THROW_IE_EXCEPTION << "Input and output ports should be equal";
    }
});

REG_CONVERTER_FOR(MVN, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    layer.getParameters()["across_channels"] = cnnLayer->GetParamAsBool("across_channels", 0);
    layer.getParameters()["normalize_variance"] = cnnLayer->GetParamAsBool("normalize_variance", 0);
    layer.getParameters()["eps"] = cnnLayer->GetParamAsFloat("eps", 0);
});