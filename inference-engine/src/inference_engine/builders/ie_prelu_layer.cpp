// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_prelu_layer.hpp>
#include <details/caseless.hpp>

#include <string>

using namespace InferenceEngine;

Builder::PReLULayer::PReLULayer(const std::string& name): LayerFragment("PReLU", name) {
    getLayer().getOutputPorts().resize(1);
    getLayer().getInputPorts().resize(1);
    setChannelShared(false);
}

Builder::PReLULayer::PReLULayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "PReLU"))
        THROW_IE_EXCEPTION << "Cannot create PReLULayer decorator for layer " << getLayer().getType();
}

Builder::PReLULayer& Builder::PReLULayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}

const Port& Builder::PReLULayer::getPort() const {
    return getLayer().getOutputPorts()[0];
}

Builder::PReLULayer& Builder::PReLULayer::setPort(const Port &port) {
    getLayer().getOutputPorts()[0] = port;
    getLayer().getInputPorts()[0] = port;
    return *this;
}

bool Builder::PReLULayer::getChannelShared() const {
    return getLayer().getParameters()["channel_shared"].asBool();
}
Builder::PReLULayer& Builder::PReLULayer::setChannelShared(bool flag) {
    getLayer().getParameters()["channel_shared"] = flag ? 1 : 0;
    return *this;
}

Builder::PReLULayer& Builder::PReLULayer::setWeights(const Blob::CPtr& weights) {
    getLayer().addConstantData("weights", weights);
    return *this;
}
