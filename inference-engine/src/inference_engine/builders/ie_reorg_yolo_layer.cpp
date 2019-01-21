// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_reorg_yolo_layer.hpp>
#include <details/caseless.hpp>
#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::ReorgYoloLayer::ReorgYoloLayer(const std::string& name): LayerFragment("ReorgYolo", name) {
    getLayer().getInputPorts().resize(1);
    getLayer().getOutputPorts().resize(1);
}

Builder::ReorgYoloLayer::ReorgYoloLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "ReorgYolo"))
        THROW_IE_EXCEPTION << "Cannot create ReorgYoloLayer decorator for layer " << getLayer().getType();
}

Builder::ReorgYoloLayer& Builder::ReorgYoloLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}
const Port& Builder::ReorgYoloLayer::getInputPort() const {
    return getLayer().getInputPorts()[0];
}
Builder::ReorgYoloLayer& Builder::ReorgYoloLayer::setInputPort(const Port& port) {
    getLayer().getInputPorts()[0] = port;
    return *this;
}
const Port& Builder::ReorgYoloLayer::getOutputPort() const {
    return getLayer().getOutputPorts()[0];
}
Builder::ReorgYoloLayer& Builder::ReorgYoloLayer::setOutputPort(const Port& port) {
    getLayer().getOutputPorts()[0] = port;
    return *this;
}
int Builder::ReorgYoloLayer::getStride() const {
    return getLayer().getParameters()["stride"].asInt();
}
Builder::ReorgYoloLayer& Builder::ReorgYoloLayer::setStride(int stride) {
    getLayer().getParameters()["stride"] = stride;
    return *this;
}

