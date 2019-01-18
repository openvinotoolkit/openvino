// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_argmax_layer.hpp>
#include <details/caseless.hpp>

#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::ArgMaxLayer::ArgMaxLayer(const std::string& name): LayerFragment("ArgMax", name) {
    getLayer().getOutputPorts().resize(1);
    getLayer().getInputPorts().resize(1);
}

Builder::ArgMaxLayer::ArgMaxLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "ArgMax"))
        THROW_IE_EXCEPTION << "Cannot create ArgMaxLayer decorator for layer " << getLayer().getType();
}

Builder::ArgMaxLayer& Builder::ArgMaxLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}

const Port& Builder::ArgMaxLayer::getPort() const {
    return getLayer().getInputPorts()[0];
}

Builder::ArgMaxLayer& Builder::ArgMaxLayer::setPort(const Port &port) {
    getLayer().getInputPorts()[0] = port;
    getLayer().getOutputPorts()[0] = port;
    return *this;
}

int Builder::ArgMaxLayer::getAxis() const {
    return getLayer().getParameters()["axis"].asInt();
}
Builder::ArgMaxLayer& Builder::ArgMaxLayer::setAxis(int axis) {
    getLayer().getParameters()["axis"] = axis;
    return *this;
}
size_t Builder::ArgMaxLayer::getTopK() const {
    return getLayer().getParameters()["top_k"].asUInt();
}
Builder::ArgMaxLayer& Builder::ArgMaxLayer::setTopK(size_t topK) {
    getLayer().getParameters()["top_k"] = topK;
    return *this;
}
size_t Builder::ArgMaxLayer::getOutMaxVal() const {
    return getLayer().getParameters()["out_max_val"].asUInt();
}
Builder::ArgMaxLayer& Builder::ArgMaxLayer::setOutMaxVal(size_t outMaxVal) {
    if (outMaxVal > 1)
        THROW_IE_EXCEPTION << "OutMaxVal supports only 0 and 1 values.";
    getLayer().getParameters()["out_max_val"] = outMaxVal;
    return *this;
}

