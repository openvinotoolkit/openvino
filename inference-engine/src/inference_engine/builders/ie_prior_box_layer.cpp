// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_prior_box_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::PriorBoxLayer::PriorBoxLayer(const std::string& name): LayerDecorator("PriorBox", name) {
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getInputPorts().resize(2);
    setScaleAllSizes(true);
}

Builder::PriorBoxLayer::PriorBoxLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("PriorBox");
}

Builder::PriorBoxLayer::PriorBoxLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("PriorBox");
}

Builder::PriorBoxLayer& Builder::PriorBoxLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const std::vector<Port>& Builder::PriorBoxLayer::getInputPorts() const {
    return getLayer()->getInputPorts();
}

Builder::PriorBoxLayer& Builder::PriorBoxLayer::setInputPorts(const std::vector<Port> &ports) {
    if (ports.size() != 2)
        THROW_IE_EXCEPTION << "Incorrect number of inputs for PriorBox getLayer().";
    getLayer()->getInputPorts() = ports;
    return *this;
}

const Port& Builder::PriorBoxLayer::getOutputPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::PriorBoxLayer& Builder::PriorBoxLayer::setOutputPort(const Port &port) {
    getLayer()->getOutputPorts()[0] = port;
    return *this;
}

float Builder::PriorBoxLayer::getVariance() const {
    return getLayer()->getParameters().at("variance");
}
Builder::PriorBoxLayer& Builder::PriorBoxLayer::setVariance(float variance) {
    getLayer()->getParameters()["variance"] = variance;
    return *this;
}

float Builder::PriorBoxLayer::getOffset() const {
    return getLayer()->getParameters().at("offset");
}
Builder::PriorBoxLayer& Builder::PriorBoxLayer::setOffset(float offset) {
    getLayer()->getParameters()["offset"] = offset;
    return *this;
}

float Builder::PriorBoxLayer::getStep() const {
    return getLayer()->getParameters().at("step");
}
Builder::PriorBoxLayer& Builder::PriorBoxLayer::setStep(float step) {
    getLayer()->getParameters()["step"] = step;
    return *this;
}

size_t Builder::PriorBoxLayer::getMinSize() const {
    return getLayer()->getParameters().at("min_size");
}
Builder::PriorBoxLayer& Builder::PriorBoxLayer::setMinSize(size_t minSize) {
    getLayer()->getParameters()["min_size"] = minSize;
    return *this;
}
size_t Builder::PriorBoxLayer::getMaxSize() const {
    return getLayer()->getParameters().at("max_size");
}
Builder::PriorBoxLayer& Builder::PriorBoxLayer::setMaxSize(size_t maxSize) {
    getLayer()->getParameters()["max_size"] = maxSize;
    return *this;
}

bool Builder::PriorBoxLayer::getScaleAllSizes() const {
    return getLayer()->getParameters().at("scale_all_sizes");
}
Builder::PriorBoxLayer& Builder::PriorBoxLayer::setScaleAllSizes(bool flag) {
    getLayer()->getParameters()["scale_all_sizes"] = flag;
    return *this;
}

bool Builder::PriorBoxLayer::getClip() const {
    return getLayer()->getParameters().at("clip");
}
Builder::PriorBoxLayer& Builder::PriorBoxLayer::setClip(bool flag) {
    getLayer()->getParameters()["clip"] = flag;
    return *this;
}

bool Builder::PriorBoxLayer::getFlip() const {
    return getLayer()->getParameters().at("flip");
}
Builder::PriorBoxLayer& Builder::PriorBoxLayer::setFlip(bool flag) {
    getLayer()->getParameters()["flip"] = flag;
    return *this;
}

const std::vector<size_t> Builder::PriorBoxLayer::getAspectRatio() const {
    return getLayer()->getParameters().at("aspect_ratio");
}
Builder::PriorBoxLayer& Builder::PriorBoxLayer::setAspectRatio(const std::vector<size_t>& aspectRatio) {
    getLayer()->getParameters()["aspect_ratio"] = aspectRatio;
    return *this;
}

REG_CONVERTER_FOR(PriorBox, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    layer.getParameters()["flip"] = cnnLayer->GetParamAsBool("flip", false);
    layer.getParameters()["clip"] = cnnLayer->GetParamAsBool("clip", false);
    layer.getParameters()["scale_all_sizes"] = cnnLayer->GetParamAsBool("scale_all_sizes", true);
    layer.getParameters()["step"] = cnnLayer->GetParamAsFloat("step", 0);
    layer.getParameters()["offset"] = cnnLayer->GetParamAsFloat("offset");
    layer.getParameters()["variance"] = cnnLayer->GetParamAsFloats("variance", {});
    layer.getParameters()["aspect_ratio"] = cnnLayer->GetParamAsFloats("aspect_ratio", {});
    layer.getParameters()["min_size"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("min_size", 0));
    layer.getParameters()["max_size"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("max_size", 0));
});
