// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_pooling_layer.hpp>
#include <details/caseless.hpp>

#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::PoolingLayer::PoolingLayer(const std::string& name): LayerFragment("Pooling", name) {
    getLayer().getInputPorts().resize(1);
    getLayer().getOutputPorts().resize(1);
    setExcludePad(false);
    setPoolingType(PoolingType::MAX);
    setRoundingType(RoundingType::CEIL);
}

Builder::PoolingLayer::PoolingLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "Pooling"))
        THROW_IE_EXCEPTION << "Cannot create PoolingLayer decorator for layer " << getLayer().getType();

    std::string typeStr = getLayer().getParameters()["pool-method"].asString("max");
    if (typeStr == "max")
        type = MAX;
    else if (typeStr == "avg")
        type = AVG;

    typeStr = getLayer().getParameters()["rounding_type"].asString("ceil");
    if (typeStr == "ceil")
        roundingType = CEIL;
    else if (typeStr == "avg")
        roundingType = FLOOR;
}

Builder::PoolingLayer::operator Builder::Layer() const {
    Layer genLayer(getLayer());

    std::vector<size_t> l_kernel = getKernel();
    std::vector<size_t> l_paddingBegin = getPaddingsBegin();
    std::vector<size_t> l_paddingEnd = getPaddingsEnd();
    std::vector<size_t> l_strides = getStrides();

    if (l_paddingBegin.empty() && !l_kernel.empty())
        l_paddingBegin.resize(l_kernel.size(), 0);
    if (l_paddingEnd.empty() && !l_kernel.empty())
        l_paddingEnd.resize(l_kernel.size(), 0);
    if (l_strides.empty() && !l_kernel.empty())
        l_strides.resize(l_kernel.size(), 1);

    if (l_kernel.empty() || l_kernel.size() != l_paddingBegin.size() || l_kernel.size() != l_paddingEnd.size() || l_kernel.size() != l_strides.size())
        THROW_IE_EXCEPTION << genLayer.getType() << " node " << genLayer.getName() << " contains incorrect parameters!";

    genLayer.getParameters()["kernel"] = l_kernel;
    genLayer.getParameters()["strides"] = l_strides;
    genLayer.getParameters()["pads_begin"] = l_paddingBegin;
    genLayer.getParameters()["pads_end"] = l_paddingEnd;
    return genLayer;
}

Builder::PoolingLayer &Builder::PoolingLayer::setName(const std::string &name) {
    getLayer().getName() = name;
    return *this;
}

const Port& Builder::PoolingLayer::getInputPort() const {
    return getLayer().getInputPorts()[0];
}

Builder::PoolingLayer& Builder::PoolingLayer::setInputPort(const Port& port) {
    getLayer().getInputPorts()[0] = port;
    return *this;
}

const Port& Builder::PoolingLayer::getOutputPort() const {
    return getLayer().getOutputPorts()[0];
}

Builder::PoolingLayer& Builder::PoolingLayer::setOutputPort(const Port& port) {
    getLayer().getOutputPorts()[0] = port;
    return *this;
}

const std::vector<size_t> Builder::PoolingLayer::getKernel() const {
    return uInts2size_t(getLayer().getParameters()["kernel"].asUInts({}));
}
Builder::PoolingLayer& Builder::PoolingLayer::setKernel(const std::vector<size_t>& kernel) {
    getLayer().getParameters()["kernel"] = kernel;
    return *this;
}

const std::vector<size_t> Builder::PoolingLayer::getStrides() const {
    return uInts2size_t(getLayer().getParameters()["strides"].asUInts({}));
}
Builder::PoolingLayer& Builder::PoolingLayer::setStrides(const std::vector<size_t>& strides) {
    getLayer().getParameters()["strides"] = strides;
    return *this;
}

const std::vector<size_t> Builder::PoolingLayer::getPaddingsBegin() const {
    return uInts2size_t(getLayer().getParameters()["pads_begin"].asUInts({}));
}
Builder::PoolingLayer& Builder::PoolingLayer::setPaddingsBegin(const std::vector<size_t>& paddings) {
    getLayer().getParameters()["pads_begin"] = paddings;
    return *this;
}

const std::vector<size_t> Builder::PoolingLayer::getPaddingsEnd() const {
    return uInts2size_t(getLayer().getParameters()["pads_end"].asUInts({}));
}
Builder::PoolingLayer& Builder::PoolingLayer::setPaddingsEnd(const std::vector<size_t>& paddings) {
    getLayer().getParameters()["pads_end"] = paddings;
    return *this;
}

Builder::PoolingLayer::PoolingType Builder::PoolingLayer::getPoolingType() const {
    return type;
}
Builder::PoolingLayer& Builder::PoolingLayer::setPoolingType(Builder::PoolingLayer::PoolingType type) {
    this->type = type;
    std::string typeStr;
    switch (type) {
        case MAX:
            typeStr = "max";
            break;
        case AVG:
            typeStr = "avg";
            break;
    }
    getLayer().getParameters()["pool-method"] = typeStr;
    return *this;
}

Builder::PoolingLayer::RoundingType Builder::PoolingLayer::getRoundingType() const {
    return roundingType;
}
Builder::PoolingLayer& Builder::PoolingLayer::setRoundingType(Builder::PoolingLayer::RoundingType type) {
    roundingType = type;
    std::string typeStr;
    switch (type) {
        case CEIL:
            typeStr = "ceil";
            break;
        case FLOOR:
            typeStr = "floor";
            break;
    }
    getLayer().getParameters()["rounding_type"] = typeStr;
    return *this;
}

bool Builder::PoolingLayer::getExcludePad() const {
    return getLayer().getParameters()["exclude-pad"].asBool();
}

Builder::PoolingLayer& Builder::PoolingLayer::setExcludePad(bool exclude) {
    getLayer().getParameters()["exclude-pad"] = exclude;
    return *this;
}


void Builder::PoolingLayer::validate(const Layer& layer) {
    Layer poolLayer = layer;
    Builder::PoolingLayer poolBuilder(poolLayer);
    std::vector<size_t> l_kernel = poolBuilder.getKernel();
    // WA for old IRs
    if (l_kernel.empty() && layer.getParameters().find("kernel-x") != layer.getParameters().end() &&
        layer.getParameters().find("kernel-y") != layer.getParameters().end())
        return;
    std::vector<size_t> l_paddingBegin = poolBuilder.getPaddingsBegin();
    std::vector<size_t> l_paddingEnd = poolBuilder.getPaddingsEnd();
    std::vector<size_t> l_strides = poolBuilder.getStrides();

    if (l_paddingBegin.empty() && !l_kernel.empty())
        l_paddingBegin.resize(l_kernel.size(), 0);
    if (l_paddingEnd.empty() && !l_kernel.empty())
        l_paddingEnd.resize(l_kernel.size(), 0);
    if (l_strides.empty() && !l_kernel.empty())
        l_strides.resize(l_kernel.size(), 1);

    if (l_kernel.empty() || l_kernel.size() != l_paddingBegin.size() || l_kernel.size() != l_paddingEnd.size() || l_kernel.size() != l_strides.size())
        THROW_IE_EXCEPTION << layer.getType() << " node " << layer.getName() << " contains incorrect parameters!";
}

REG_VALIDATOR_FOR(Pooling, Builder::PoolingLayer::validate);
