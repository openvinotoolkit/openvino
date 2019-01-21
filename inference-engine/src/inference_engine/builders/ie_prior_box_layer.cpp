// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_prior_box_layer.hpp>
#include <details/caseless.hpp>

#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::PriorBoxLayer::PriorBoxLayer(const std::string& name): LayerFragment("PriorBox", name) {
    getLayer().getOutputPorts().resize(1);
    getLayer().getInputPorts().resize(2);
    setScaleAllSizes(true);
}

Builder::PriorBoxLayer::PriorBoxLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "PriorBox"))
        THROW_IE_EXCEPTION << "Cannot create PriorBoxLayer decorator for layer " << getLayer().getType();
}

Builder::PriorBoxLayer& Builder::PriorBoxLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}

const std::vector<Port>& Builder::PriorBoxLayer::getInputPorts() const {
    return getLayer().getInputPorts();
}

Builder::PriorBoxLayer& Builder::PriorBoxLayer::setInputPorts(const std::vector<Port> &ports) {
    if (ports.size() != 2)
        THROW_IE_EXCEPTION << "Incorrect number of inputs for PriorBox layer.";
    getLayer().getInputPorts() = ports;
    return *this;
}

const Port& Builder::PriorBoxLayer::getOutputPort() const {
    return getLayer().getOutputPorts()[0];
}

Builder::PriorBoxLayer& Builder::PriorBoxLayer::setOutputPort(const Port &port) {
    getLayer().getOutputPorts()[0] = port;
    return *this;
}

float Builder::PriorBoxLayer::getVariance() const {
    return getLayer().getParameters()["variance"].asFloat();
}
Builder::PriorBoxLayer& Builder::PriorBoxLayer::setVariance(float variance) {
    getLayer().getParameters()["variance"] = variance;
    return *this;
}

float Builder::PriorBoxLayer::getOffset() const {
    return getLayer().getParameters()["offset"].asFloat();
}
Builder::PriorBoxLayer& Builder::PriorBoxLayer::setOffset(float offset) {
    getLayer().getParameters()["offset"] = offset;
    return *this;
}

float Builder::PriorBoxLayer::getStep() const {
    return getLayer().getParameters()["step"].asFloat();
}
Builder::PriorBoxLayer& Builder::PriorBoxLayer::setStep(float step) {
    getLayer().getParameters()["step"] = step;
    return *this;
}

size_t Builder::PriorBoxLayer::getMinSize() const {
    return getLayer().getParameters()["min_size"].asUInt();
}
Builder::PriorBoxLayer& Builder::PriorBoxLayer::setMinSize(size_t minSize) {
    getLayer().getParameters()["min_size"] = minSize;
    return *this;
}
size_t Builder::PriorBoxLayer::getMaxSize() const {
    return getLayer().getParameters()["max_size"].asUInt();
}
Builder::PriorBoxLayer& Builder::PriorBoxLayer::setMaxSize(size_t maxSize) {
    getLayer().getParameters()["max_size"] = maxSize;
    return *this;
}

bool Builder::PriorBoxLayer::getScaleAllSizes() const {
    return getLayer().getParameters()["scale_all_sizes"].asBool(true);
}
Builder::PriorBoxLayer& Builder::PriorBoxLayer::setScaleAllSizes(bool flag) {
    getLayer().getParameters()["scale_all_sizes"] = flag;
    return *this;
}

bool Builder::PriorBoxLayer::getClip() const {
    return getLayer().getParameters()["clip"].asBool();
}
Builder::PriorBoxLayer& Builder::PriorBoxLayer::setClip(bool flag) {
    getLayer().getParameters()["clip"] = flag;
    return *this;
}

bool Builder::PriorBoxLayer::getFlip() const {
    return getLayer().getParameters()["flip"].asBool();
}
Builder::PriorBoxLayer& Builder::PriorBoxLayer::setFlip(bool flag) {
    getLayer().getParameters()["flip"] = flag;
    return *this;
}

const std::vector<size_t> Builder::PriorBoxLayer::getAspectRatio() const {
    return uInts2size_t(getLayer().getParameters()["aspect_ratio"].asUInts());
}
Builder::PriorBoxLayer& Builder::PriorBoxLayer::setAspectRatio(const std::vector<size_t>& aspectRatio) {
    getLayer().getParameters()["aspect_ratio"] = aspectRatio;
    return *this;
}
