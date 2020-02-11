// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_reorg_yolo_layer.hpp>
#include <ie_cnn_layer_builder.h>
#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::ReorgYoloLayer::ReorgYoloLayer(const std::string& name): LayerDecorator("ReorgYolo", name) {
    getLayer()->getInputPorts().resize(1);
    getLayer()->getOutputPorts().resize(1);
}

Builder::ReorgYoloLayer::ReorgYoloLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("ReorgYolo");
}

Builder::ReorgYoloLayer::ReorgYoloLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("ReorgYolo");
}

Builder::ReorgYoloLayer& Builder::ReorgYoloLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}
const Port& Builder::ReorgYoloLayer::getInputPort() const {
    return getLayer()->getInputPorts()[0];
}
Builder::ReorgYoloLayer& Builder::ReorgYoloLayer::setInputPort(const Port& port) {
    getLayer()->getInputPorts()[0] = port;
    return *this;
}
const Port& Builder::ReorgYoloLayer::getOutputPort() const {
    return getLayer()->getOutputPorts()[0];
}
Builder::ReorgYoloLayer& Builder::ReorgYoloLayer::setOutputPort(const Port& port) {
    getLayer()->getOutputPorts()[0] = port;
    return *this;
}
int Builder::ReorgYoloLayer::getStride() const {
    return getLayer()->getParameters().at("stride");
}
Builder::ReorgYoloLayer& Builder::ReorgYoloLayer::setStride(int stride) {
    getLayer()->getParameters()["stride"] = stride;
    return *this;
}

REG_CONVERTER_FOR(ReorgYolo, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    layer.getParameters()["stride"] = cnnLayer->GetParamAsInt("stride", 0);
});