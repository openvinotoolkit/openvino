// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_softmax_layer.hpp>
#include <details/caseless.hpp>

#include <string>

using namespace InferenceEngine;

Builder::SoftMaxLayer::SoftMaxLayer(const std::string& name): LayerFragment("SoftMax", name) {
    getLayer().getOutputPorts().resize(1);
    getLayer().getInputPorts().resize(1);
    setAxis(1);
}

Builder::SoftMaxLayer::SoftMaxLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "SoftMax"))
        THROW_IE_EXCEPTION << "Cannot create SoftMaxLayer decorator for layer " << getLayer().getType();
}

Builder::SoftMaxLayer& Builder::SoftMaxLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}

const Port& Builder::SoftMaxLayer::getPort() const {
    return getLayer().getOutputPorts()[0];
}

Builder::SoftMaxLayer& Builder::SoftMaxLayer::setPort(const Port &port) {
    getLayer().getOutputPorts()[0] = port;
    getLayer().getInputPorts()[0] = port;
    return *this;
}

size_t Builder::SoftMaxLayer::getAxis() const {
    return getLayer().getParameters()["axis"].asUInt();
}

Builder::SoftMaxLayer& Builder::SoftMaxLayer::setAxis(size_t axis) {
    getLayer().getParameters()["axis"] = axis;
    return *this;
}
