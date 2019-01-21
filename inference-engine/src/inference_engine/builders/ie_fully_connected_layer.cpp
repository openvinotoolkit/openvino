// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_fully_connected_layer.hpp>
#include <details/caseless.hpp>

#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::FullyConnectedLayer::FullyConnectedLayer(const std::string& name): LayerFragment("FullyConnected", name) {
    getLayer().getInputPorts().resize(1);
    getLayer().getOutputPorts().resize(1);
    getLayer().getParameters()["out-size"] = 0;
}

Builder::FullyConnectedLayer::FullyConnectedLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "FullyConnected"))
        THROW_IE_EXCEPTION << "Cannot create FullyConnectedLayer decorator for layer " << getLayer().getType();
}

Builder::FullyConnectedLayer &Builder::FullyConnectedLayer::setName(const std::string &name) {
    getLayer().getName() = name;
    return *this;
}

Builder::FullyConnectedLayer& Builder::FullyConnectedLayer::setWeights(const Blob::CPtr& weights) {
    getLayer().addConstantData("weights", weights);
    return *this;
}
Builder::FullyConnectedLayer& Builder::FullyConnectedLayer::setBiases(const Blob::CPtr& biases) {
    getLayer().addConstantData("biases", biases);
    return *this;
}

const Port& Builder::FullyConnectedLayer::getInputPort() const {
    return getLayer().getInputPorts()[0];
}

Builder::FullyConnectedLayer& Builder::FullyConnectedLayer::setInputPort(const Port& port) {
    getLayer().getInputPorts()[0] = port;
    return *this;
}

const Port& Builder::FullyConnectedLayer::getOutputPort() const {
    return getLayer().getOutputPorts()[0];
}

Builder::FullyConnectedLayer& Builder::FullyConnectedLayer::setOutputPort(const Port& port) {
    getLayer().getOutputPorts()[0] = port;
    return *this;
}

size_t Builder::FullyConnectedLayer::getOutputNum() const {
    return getLayer().getParameters()["out-size"].asUInt();
}
Builder::FullyConnectedLayer& Builder::FullyConnectedLayer::setOutputNum(size_t outNum) {
    getLayer().getParameters()["out-size"] = outNum;
    return *this;
}
