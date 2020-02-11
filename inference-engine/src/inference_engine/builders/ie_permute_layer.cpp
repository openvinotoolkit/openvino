// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_permute_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <string>
#include <vector>

using namespace InferenceEngine;

Builder::PermuteLayer::PermuteLayer(const std::string& name): LayerDecorator("Permute", name) {
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getInputPorts().resize(1);
}

Builder::PermuteLayer::PermuteLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("Permute");
}

Builder::PermuteLayer::PermuteLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("Permute");
}

Builder::PermuteLayer& Builder::PermuteLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::PermuteLayer::getOutputPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::PermuteLayer& Builder::PermuteLayer::setOutputPort(const Port &port) {
    getLayer()->getOutputPorts()[0] = port;
    return *this;
}

const Port& Builder::PermuteLayer::getInputPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::PermuteLayer& Builder::PermuteLayer::setInputPort(const Port &port) {
    getLayer()->getOutputPorts()[0] = port;
    return *this;
}

const std::vector<size_t> Builder::PermuteLayer::getOrder() const {
    return getLayer()->getParameters().at("order");
}
Builder::PermuteLayer& Builder::PermuteLayer::setOrder(const std::vector<size_t>& ratios) {
    getLayer()->getParameters()["order"] = ratios;
    return *this;
}

REG_CONVERTER_FOR(Permute, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    std::vector<unsigned int> tmp = cnnLayer->GetParamAsUInts("order");
    layer.getParameters()["order"] = std::vector<size_t>(tmp.size());
    for (size_t i = 0; i < tmp.size(); ++i) {
        layer.getParameters()["order"].as<std::vector<size_t>>()[i] = static_cast<size_t>(tmp[i]);
    }
});