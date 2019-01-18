// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_batch_normalization_layer.hpp>
#include <details/caseless.hpp>

#include <string>

using namespace InferenceEngine;

Builder::BatchNormalizationLayer::BatchNormalizationLayer(const std::string& name): LayerFragment("BatchNormalization", name) {
    getLayer().getOutputPorts().resize(1);
    getLayer().getInputPorts().resize(1);
    setEpsilon(0.00000001f);
}

Builder::BatchNormalizationLayer::BatchNormalizationLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "BatchNormalization"))
        THROW_IE_EXCEPTION << "Cannot create BatchNormalizationLayer decorator for layer " << getLayer().getType();
}

Builder::BatchNormalizationLayer& Builder::BatchNormalizationLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}

const Port& Builder::BatchNormalizationLayer::getPort() const {
    return getLayer().getOutputPorts()[0];
}

Builder::BatchNormalizationLayer& Builder::BatchNormalizationLayer::setPort(const Port &port) {
    getLayer().getOutputPorts()[0] = port;
    getLayer().getInputPorts()[0] = port;
    return *this;
}

Builder::BatchNormalizationLayer& Builder::BatchNormalizationLayer::setWeights(const Blob::CPtr& weights) {
    getLayer().addConstantData("weights", weights);
    return *this;
}
Builder::BatchNormalizationLayer& Builder::BatchNormalizationLayer::setBiases(const Blob::CPtr& biases) {
    getLayer().addConstantData("biases", biases);
    return *this;
}

float Builder::BatchNormalizationLayer::getEpsilon() const {
    return getLayer().getParameters()["epsilon"].asFloat();
}
Builder::BatchNormalizationLayer& Builder::BatchNormalizationLayer::setEpsilon(float eps) {
    getLayer().getParameters()["epsilon"] = eps;
    return *this;
}

void Builder::BatchNormalizationLayer::validate(const Layer& layer)  {
    auto weightsIt = layer.getConstantData().find("weights");
    auto biasesIt = layer.getConstantData().find("biases");
    bool valid = weightsIt != layer.getConstantData().end() &&
            biasesIt != layer.getConstantData().end() &&
            weightsIt->second != nullptr &&
            weightsIt->second->cbuffer() != nullptr &&
            biasesIt->second != nullptr &&
            biasesIt->second->cbuffer() != nullptr;
    if (!valid)
        THROW_IE_EXCEPTION << "Cannot create BatchNotmalization layer! Weights and biases are required!";
}

REG_VALIDATOR_FOR(BatchNormalization,  Builder::BatchNormalizationLayer::validate);