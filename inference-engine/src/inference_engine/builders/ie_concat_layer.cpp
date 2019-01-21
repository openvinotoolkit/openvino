// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_concat_layer.hpp>
#include <details/caseless.hpp>

#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::ConcatLayer::ConcatLayer(const std::string& name): LayerFragment("Concat", name) {
    getLayer().getOutputPorts().resize(1);
    setAxis(1);
}

Builder::ConcatLayer::ConcatLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "Concat"))
        THROW_IE_EXCEPTION << "Cannot create ConcatLayer decorator for layer " << getLayer().getType();
}

Builder::ConcatLayer& Builder::ConcatLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}

const Port& Builder::ConcatLayer::getOutputPort() const {
    return getLayer().getOutputPorts()[0];
}

Builder::ConcatLayer& Builder::ConcatLayer::setOutputPort(const Port &port) {
    getLayer().getOutputPorts()[0] = port;
    return *this;
}

const std::vector<Port>& Builder::ConcatLayer::getInputPorts() const {
    return getLayer().getInputPorts();
}

Builder::ConcatLayer& Builder::ConcatLayer::setInputPorts(const std::vector<Port>& ports) {
    getLayer().getInputPorts() = ports;
    return *this;
}

size_t Builder::ConcatLayer::getAxis() const {
    return getLayer().getParameters()["axis"].asUInt();
}

Builder::ConcatLayer& Builder::ConcatLayer::setAxis(size_t axis) {
    getLayer().getParameters()["axis"] = axis;
    return *this;
}
