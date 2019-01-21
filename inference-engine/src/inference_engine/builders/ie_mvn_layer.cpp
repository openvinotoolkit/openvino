// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_mvn_layer.hpp>
#include <details/caseless.hpp>

#include <string>

using namespace InferenceEngine;

Builder::MVNLayer::MVNLayer(const std::string& name): LayerFragment("MVN", name) {
    getLayer().getOutputPorts().resize(1);
    getLayer().getInputPorts().resize(1);
    setEpsilon(9.999999717180685e-10f);
    setNormalize(true);
    setAcrossChannels(true);
}

Builder::MVNLayer::MVNLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "MVN"))
        THROW_IE_EXCEPTION << "Cannot create MVNLayer decorator for layer " << getLayer().getType();
}

Builder::MVNLayer& Builder::MVNLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}

const Port& Builder::MVNLayer::getPort() const {
    return getLayer().getOutputPorts()[0];
}

Builder::MVNLayer& Builder::MVNLayer::setPort(const Port &port) {
    getLayer().getOutputPorts()[0] = port;
    getLayer().getInputPorts()[0] = port;
    return *this;
}

bool Builder::MVNLayer::getAcrossChannels() const {
    return getLayer().getParameters()["across_channels"].asBool(true);
}
Builder::MVNLayer& Builder::MVNLayer::setAcrossChannels(bool flag) {
    getLayer().getParameters()["across_channels"] = flag ? 1 : 0;
    return *this;
}
bool Builder::MVNLayer::getNormalize() const {
    return getLayer().getParameters()["normalize_variance"].asBool(true);
}
Builder::MVNLayer& Builder::MVNLayer::setNormalize(bool flag) {
    getLayer().getParameters()["normalize_variance"] = flag ? 1 : 0;
    return *this;
}
float Builder::MVNLayer::getEpsilon() const {
    return getLayer().getParameters()["eps"].asFloat();
}
Builder::MVNLayer& Builder::MVNLayer::setEpsilon(float eps) {
    getLayer().getParameters()["eps"] = eps;
    return *this;
}
