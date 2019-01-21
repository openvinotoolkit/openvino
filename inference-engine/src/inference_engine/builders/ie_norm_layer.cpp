// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_norm_layer.hpp>
#include <details/caseless.hpp>

#include <string>

using namespace InferenceEngine;

Builder::NormLayer::NormLayer(const std::string& name): LayerFragment("Norm", name) {
    getLayer().getOutputPorts().resize(1);
    getLayer().getInputPorts().resize(1);
    setAcrossMaps(false);
    setSize(0);
    setAlpha(0);
    setBeta(0);
}

Builder::NormLayer::NormLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "Norm"))
        THROW_IE_EXCEPTION << "Cannot create NormLayer decorator for layer " << getLayer().getType();
}

Builder::NormLayer& Builder::NormLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}

const Port& Builder::NormLayer::getPort() const {
    return getLayer().getOutputPorts()[0];
}

Builder::NormLayer& Builder::NormLayer::setPort(const Port &port) {
    getLayer().getOutputPorts()[0] = port;
    getLayer().getInputPorts()[0] = port;
    return *this;
}

size_t Builder::NormLayer::getSize() const {
    return getLayer().getParameters()["local-size"].asUInt();
}

Builder::NormLayer& Builder::NormLayer::setSize(size_t size) {
    getLayer().getParameters()["local-size"] = size;
    return *this;
}

float Builder::NormLayer::getAlpha() const {
    return getLayer().getParameters()["alpha"].asFloat();
}

Builder::NormLayer& Builder::NormLayer::setAlpha(float alpha) {
    getLayer().getParameters()["alpha"] = alpha;
    return *this;
}

float Builder::NormLayer::getBeta() const {
    return getLayer().getParameters()["beta"].asFloat();
}

Builder::NormLayer& Builder::NormLayer::setBeta(float beta) {
    getLayer().getParameters()["beta"] = beta;
    return *this;
}

bool Builder::NormLayer::getAcrossMaps() const {
    return getLayer().getParameters()["region"].asString() == "across";
}

Builder::NormLayer& Builder::NormLayer::setAcrossMaps(bool acrossMap)  {
    std::string value = acrossMap ? "across" : "same";
    getLayer().getParameters()["region"] = value;
    return *this;
}

Builder::NormLayer::NormType Builder::NormLayer::getRegion() const {
    return getAcrossMaps() ? Builder::NormLayer::NormType::ACROSS_CHANNELS :
                             Builder::NormLayer::NormType::WITHIN_CHANNEL;
}
Builder::NormLayer& Builder::NormLayer::setRegion(Builder::NormLayer::NormType type) {
    setAcrossMaps(type);
    return *this;
}
