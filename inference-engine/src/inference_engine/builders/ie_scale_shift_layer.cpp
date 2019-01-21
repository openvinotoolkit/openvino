// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_scale_shift_layer.hpp>
#include <details/caseless.hpp>

#include <string>

using namespace InferenceEngine;

Builder::ScaleShiftLayer::ScaleShiftLayer(const std::string& name): LayerFragment("ScaleShift", name) {
    getLayer().getOutputPorts().resize(1);
    getLayer().getInputPorts().resize(1);
}

Builder::ScaleShiftLayer::ScaleShiftLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "ScaleShift"))
        THROW_IE_EXCEPTION << "Cannot create ScaleShiftLayer decorator for layer " << getLayer().getType();
}

Builder::ScaleShiftLayer& Builder::ScaleShiftLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}

const Port& Builder::ScaleShiftLayer::getPort() const {
    return getLayer().getOutputPorts()[0];
}

Builder::ScaleShiftLayer& Builder::ScaleShiftLayer::setPort(const Port &port) {
    getLayer().getOutputPorts()[0] = port;
    getLayer().getInputPorts()[0] = port;
    return *this;
}

Builder::ScaleShiftLayer& Builder::ScaleShiftLayer::setWeights(const Blob::CPtr& weights) {
    getLayer().addConstantData("weights", weights);
    return *this;
}
Builder::ScaleShiftLayer& Builder::ScaleShiftLayer::setBiases(const Blob::CPtr& biases) {
    getLayer().addConstantData("biases", biases);
    return *this;
}
