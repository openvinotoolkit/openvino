// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_normalize_layer.hpp>
#include <details/caseless.hpp>

#include <string>

using namespace InferenceEngine;

Builder::NormalizeLayer::NormalizeLayer(const std::string& name): LayerFragment("Normalize", name) {
    getLayer().getOutputPorts().resize(1);
    getLayer().getInputPorts().resize(1);
    setAcrossMaps(false);
    setChannelShared(false);
    setEpsilon(0.0000001f);
}

Builder::NormalizeLayer::NormalizeLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "Normalize"))
        THROW_IE_EXCEPTION << "Cannot create NormalizeLayer decorator for layer " << getLayer().getType();
}

Builder::NormalizeLayer& Builder::NormalizeLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}

const Port& Builder::NormalizeLayer::getPort() const {
    return getLayer().getOutputPorts()[0];
}

Builder::NormalizeLayer& Builder::NormalizeLayer::setPort(const Port &port) {
    getLayer().getOutputPorts()[0] = port;
    getLayer().getInputPorts()[0] = port;
    return *this;
}

bool Builder::NormalizeLayer::getAcrossMaps() const {
    return getLayer().getParameters()["region"].asBool();
}

Builder::NormalizeLayer& Builder::NormalizeLayer::setAcrossMaps(bool acrossMap)  {
    getLayer().getParameters()["region"] = acrossMap ? 1 : 0;
    return *this;
}

bool Builder::NormalizeLayer::getChannelShared() const {
    return getLayer().getParameters()["channel_shared"].asBool();
}

Builder::NormalizeLayer& Builder::NormalizeLayer::setChannelShared(bool channelShared)  {
    getLayer().getParameters()["channel_shared"] = channelShared ? 1 : 0;
    return *this;
}

float Builder::NormalizeLayer::getEpsilon() const {
    return getLayer().getParameters()["eps"].asFloat();
}

Builder::NormalizeLayer& Builder::NormalizeLayer::setEpsilon(float eps) {
    getLayer().getParameters()["eps"] = eps;
    return *this;
}
