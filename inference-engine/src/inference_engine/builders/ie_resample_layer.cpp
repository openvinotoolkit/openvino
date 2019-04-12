// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_resample_layer.hpp>
#include <ie_cnn_layer_builder.h>
#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::ResampleLayer::ResampleLayer(const std::string& name): LayerDecorator("Resample", name) {
    getLayer()->getInputPorts().resize(1);
    getLayer()->getOutputPorts().resize(1);
}

Builder::ResampleLayer::ResampleLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("Resample");
}

Builder::ResampleLayer::ResampleLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("Resample");
}

Builder::ResampleLayer& Builder::ResampleLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}
const Port& Builder::ResampleLayer::getInputPort() const {
    return getLayer()->getInputPorts()[0];
}
Builder::ResampleLayer& Builder::ResampleLayer::setInputPort(const Port& port) {
    getLayer()->getInputPorts()[0] = port;
    return *this;
}
const Port& Builder::ResampleLayer::getOutputPort() const {
    return getLayer()->getOutputPorts()[0];
}
Builder::ResampleLayer& Builder::ResampleLayer::setOutputPort(const Port& port) {
    getLayer()->getOutputPorts()[0] = port;
    return *this;
}

const std::string &Builder::ResampleLayer::getResampleType() const {
    return getLayer()->getParameters().at("type");
}

Builder::ResampleLayer &Builder::ResampleLayer::setResampleType(const std::string &type) {
    getLayer()->getParameters()["type"] = type;
    return *this;
}

bool Builder::ResampleLayer::getAntialias() const {
    return getLayer()->getParameters().at("antialias");
}

Builder::ResampleLayer &Builder::ResampleLayer::setAntialias(bool antialias) {
    getLayer()->getParameters()["antialias"] = antialias;
    return *this;
}

float Builder::ResampleLayer::getFactor() const {
    return getLayer()->getParameters().at("factor");
}

Builder::ResampleLayer &Builder::ResampleLayer::setFactor(float factor) {
    getLayer()->getParameters()["factor"] = factor;
    return *this;
}

size_t Builder::ResampleLayer::getWidth() const {
    return getLayer()->getParameters().at("width");
}

Builder::ResampleLayer &Builder::ResampleLayer::setWidth(size_t width) {
    getLayer()->getParameters()["width"] = width;
    return *this;
}

size_t Builder::ResampleLayer::getHeight() const {
    return getLayer()->getParameters().at("height");
}

Builder::ResampleLayer &Builder::ResampleLayer::setHeight(size_t height) {
    getLayer()->getParameters()["height"] = height;
    return *this;
}

REG_CONVERTER_FOR(Resample, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    layer.getParameters()["height"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("height", 0));
    layer.getParameters()["width"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("width", 0));
    layer.getParameters()["factor"] = cnnLayer->GetParamAsFloat("factor", 0);
    layer.getParameters()["antialias"] = cnnLayer->GetParamAsBool("antialias", false);
    layer.getParameters()["type"] = cnnLayer->GetParamAsString("type");
});