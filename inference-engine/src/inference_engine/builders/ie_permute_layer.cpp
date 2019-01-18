// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_permute_layer.hpp>
#include <details/caseless.hpp>

#include <string>
#include <vector>

using namespace InferenceEngine;

Builder::PermuteLayer::PermuteLayer(const std::string& name): LayerFragment("Permute", name) {
    getLayer().getOutputPorts().resize(1);
    getLayer().getInputPorts().resize(1);
}

Builder::PermuteLayer::PermuteLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "Permute"))
        THROW_IE_EXCEPTION << "Cannot create PermuteLayer decorator for layer " << getLayer().getType();
}

Builder::PermuteLayer& Builder::PermuteLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}

const Port& Builder::PermuteLayer::getOutputPort() const {
    return getLayer().getOutputPorts()[0];
}

Builder::PermuteLayer& Builder::PermuteLayer::setOutputPort(const Port &port) {
    getLayer().getOutputPorts()[0] = port;
    return *this;
}

const Port& Builder::PermuteLayer::getInputPort() const {
    return getLayer().getOutputPorts()[0];
}

Builder::PermuteLayer& Builder::PermuteLayer::setInputPort(const Port &port) {
    getLayer().getOutputPorts()[0] = port;
    return *this;
}

const std::vector<size_t> Builder::PermuteLayer::getOrder() const {
    return uInts2size_t(getLayer().getParameters()["order"].asUInts());
}
Builder::PermuteLayer& Builder::PermuteLayer::setOrder(const std::vector<size_t>& ratios) {
    getLayer().getParameters()["order"] = ratios;
    return *this;
}
