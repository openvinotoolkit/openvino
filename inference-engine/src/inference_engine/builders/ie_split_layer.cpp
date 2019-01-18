// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_split_layer.hpp>
#include <details/caseless.hpp>

#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::SplitLayer::SplitLayer(const std::string& name): LayerFragment("Concat", name) {
    getLayer().getInputPorts().resize(1);
    setAxis(1);
}

Builder::SplitLayer::SplitLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "Concat"))
        THROW_IE_EXCEPTION << "Cannot create SplitLayer decorator for layer " << getLayer().getType();
}

Builder::SplitLayer& Builder::SplitLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}

const Port& Builder::SplitLayer::getInputPort() const {
    return getLayer().getInputPorts()[0];
}

Builder::SplitLayer& Builder::SplitLayer::setInputPort(const Port &port) {
    getLayer().getInputPorts()[0] = port;
    return *this;
}

const std::vector<Port>& Builder::SplitLayer::getOutputPorts() const {
    return getLayer().getOutputPorts();
}

Builder::SplitLayer& Builder::SplitLayer::setOutputPorts(const std::vector<Port>& ports) {
    getLayer().getOutputPorts() = ports;
    return *this;
}

size_t Builder::SplitLayer::getAxis() const {
    return getLayer().getParameters()["axis"].asUInt();
}

Builder::SplitLayer& Builder::SplitLayer::setAxis(size_t axis) {
    getLayer().getParameters()["axis"] = axis;
    return *this;
}
