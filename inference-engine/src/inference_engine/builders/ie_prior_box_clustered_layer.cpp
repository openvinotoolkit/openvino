// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_prior_box_clustered_layer.hpp>
#include <details/caseless.hpp>

#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::PriorBoxClusteredLayer::PriorBoxClusteredLayer(const std::string& name): LayerFragment("PriorBoxClustered", name) {
    getLayer().getOutputPorts().resize(1);
    getLayer().getInputPorts().resize(2);
}

Builder::PriorBoxClusteredLayer::PriorBoxClusteredLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "PriorBoxClustered"))
        THROW_IE_EXCEPTION << "Cannot create PriorBoxClusteredLayer decorator for layer " << getLayer().getType();
}

Builder::PriorBoxClusteredLayer& Builder::PriorBoxClusteredLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}

const std::vector<Port>& Builder::PriorBoxClusteredLayer::getInputPorts() const {
    return getLayer().getInputPorts();
}

Builder::PriorBoxClusteredLayer& Builder::PriorBoxClusteredLayer::setInputPorts(const std::vector<Port> &ports) {
    if (ports.size() != 2)
        THROW_IE_EXCEPTION << "Incorrect number of inputs for PriorBoxClustered layer.";
    getLayer().getInputPorts() = ports;
    return *this;
}

const Port& Builder::PriorBoxClusteredLayer::getOutputPort() const {
    return getLayer().getOutputPorts()[0];
}

Builder::PriorBoxClusteredLayer& Builder::PriorBoxClusteredLayer::setOutputPort(const Port &port) {
    getLayer().getOutputPorts()[0] = port;
    return *this;
}

float Builder::PriorBoxClusteredLayer::getVariance() const {
    return getLayer().getParameters()["variance"].asFloat();
}
Builder::PriorBoxClusteredLayer& Builder::PriorBoxClusteredLayer::setVariance(float variance) {
    getLayer().getParameters()["variance"] = variance;
    return *this;
}

float Builder::PriorBoxClusteredLayer::getOffset() const {
    return getLayer().getParameters()["offset"].asFloat();
}
Builder::PriorBoxClusteredLayer& Builder::PriorBoxClusteredLayer::setOffset(float offset) {
    getLayer().getParameters()["offset"] = offset;
    return *this;
}

float Builder::PriorBoxClusteredLayer::getWidth() const {
    return getLayer().getParameters()["width"].asFloat();
}
Builder::PriorBoxClusteredLayer& Builder::PriorBoxClusteredLayer::setWidth(float width) {
    getLayer().getParameters()["width"] = width;
    return *this;
}

float Builder::PriorBoxClusteredLayer::getHeight() const {
    return getLayer().getParameters()["height"].asFloat();
}
Builder::PriorBoxClusteredLayer& Builder::PriorBoxClusteredLayer::setHeight(float height) {
    getLayer().getParameters()["height"] = height;
    return *this;
}

const std::vector<float> Builder::PriorBoxClusteredLayer::getSteps() const {
    return {getLayer().getParameters()["step_h"].asFloat(), getLayer().getParameters()["step_w"].asFloat()};
}
Builder::PriorBoxClusteredLayer& Builder::PriorBoxClusteredLayer::setSteps(const std::vector<float> steps) {
    if (steps.size() != 2)
        THROW_IE_EXCEPTION << "PriorBoxClusteredLayer supports sizes only for height and width dimensions!";
    getLayer().getParameters()["step_h"] = steps[0];
    getLayer().getParameters()["step_w"] = steps[1];
    return *this;
}

const std::vector<float> Builder::PriorBoxClusteredLayer::getImgSizes() const {
    return {getLayer().getParameters()["img_h"].asFloat(), getLayer().getParameters()["img_w"].asFloat()};
}
Builder::PriorBoxClusteredLayer& Builder::PriorBoxClusteredLayer::setImgSizes(const std::vector<float> sizes) {
    if (sizes.size() != 2)
        THROW_IE_EXCEPTION << "PriorBoxClusteredLayer allows to specify only height and width dimensions of an input image!";
    getLayer().getParameters()["img_h"] = sizes[0];
    getLayer().getParameters()["img_w"] = sizes[1];
    return *this;
}

float Builder::PriorBoxClusteredLayer::getStep() const {
    return getLayer().getParameters()["step"].asFloat();
}
Builder::PriorBoxClusteredLayer& Builder::PriorBoxClusteredLayer::setStep(float step) {
    getLayer().getParameters()["step"] = step;
    return *this;
}

bool Builder::PriorBoxClusteredLayer::getClip() const {
    return getLayer().getParameters()["clip"].asBool();
}
Builder::PriorBoxClusteredLayer& Builder::PriorBoxClusteredLayer::setClip(bool flag) {
    getLayer().getParameters()["clip"] = flag;
    return *this;
}

bool Builder::PriorBoxClusteredLayer::getFlip() const {
    return getLayer().getParameters()["flip"].asBool();
}
Builder::PriorBoxClusteredLayer& Builder::PriorBoxClusteredLayer::setFlip(bool flag) {
    getLayer().getParameters()["flip"] = flag;
    return *this;
}
