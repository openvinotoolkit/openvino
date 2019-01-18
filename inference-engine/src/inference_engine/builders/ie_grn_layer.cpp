// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_grn_layer.hpp>
#include <details/caseless.hpp>

#include <string>

using namespace InferenceEngine;

Builder::GRNLayer::GRNLayer(const std::string& name): LayerFragment("GRN", name) {
    getLayer().getOutputPorts().resize(1);
    getLayer().getInputPorts().resize(1);
    setBeta(0);
}

Builder::GRNLayer::GRNLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "GRN"))
        THROW_IE_EXCEPTION << "Cannot create GRNLayer decorator for layer " << getLayer().getType();
}

Builder::GRNLayer& Builder::GRNLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}

const Port& Builder::GRNLayer::getPort() const {
    return getLayer().getOutputPorts()[0];
}

Builder::GRNLayer& Builder::GRNLayer::setPort(const Port &port) {
    getLayer().getOutputPorts()[0] = port;
    getLayer().getInputPorts()[0] = port;
    return *this;
}

float Builder::GRNLayer::getBeta() const {
    return getLayer().getParameters()["beta"].asFloat();
}

Builder::GRNLayer& Builder::GRNLayer::setBeta(float beta) {
    getLayer().getParameters()["beta"] = beta;
    return *this;
}
