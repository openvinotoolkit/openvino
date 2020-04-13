// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_cnn_layer_builder.h>

#include <builders/ie_pooling_layer.hpp>
#include <string>
#include <vector>

using namespace InferenceEngine;

Builder::PoolingLayer::PoolingLayer(const std::string& name): LayerDecorator("Pooling", name) {
    getLayer()->getInputPorts().resize(1);
    getLayer()->getOutputPorts().resize(1);
    setKernel({});
    setStrides({});
    setPaddingsEnd({});
    setPaddingsBegin({});
    setExcludePad(false);
    setPoolingType(PoolingType::MAX);
    setRoundingType(RoundingType::CEIL);
}

Builder::PoolingLayer::PoolingLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("Pooling");

    std::string typeStr = getLayer()->getParameters()["pool-method"];
    if (typeStr == "max")
        type = MAX;
    else if (typeStr == "avg")
        type = AVG;

    std::string roundTypeStr = getLayer()->getParameters()["rounding_type"];
    if (roundTypeStr == "ceil")
        roundingType = CEIL;
    else if (roundTypeStr == "avg")
        roundingType = FLOOR;
}

Builder::PoolingLayer::PoolingLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("Pooling");

    const auto cLayer = static_cast<const PoolingLayer*>(this)->getLayer();

    std::string typeStr = cLayer->getParameters().at("pool-method");
    if (typeStr == "max")
        type = MAX;
    else if (typeStr == "avg")
        type = AVG;

    std::string roundTypeStr = cLayer->getParameters().at("rounding_type");
    if (roundTypeStr == "ceil")
        roundingType = CEIL;
    else if (roundTypeStr == "avg")
        roundingType = FLOOR;
}

Builder::PoolingLayer::operator Builder::Layer() const {
    Layer genLayer(*getLayer());

    std::vector<size_t> l_kernel = getKernel();
    std::vector<size_t> l_paddingBegin = getPaddingsBegin();
    std::vector<size_t> l_paddingEnd = getPaddingsEnd();
    std::vector<size_t> l_strides = getStrides();

    if (l_paddingBegin.empty() && !l_kernel.empty()) l_paddingBegin.resize(l_kernel.size(), 0);
    if (l_paddingEnd.empty() && !l_kernel.empty()) l_paddingEnd.resize(l_kernel.size(), 0);
    if (l_strides.empty() && !l_kernel.empty()) l_strides.resize(l_kernel.size(), 1);

    if (l_kernel.empty() || l_kernel.size() != l_paddingBegin.size() || l_kernel.size() != l_paddingEnd.size() ||
        l_kernel.size() != l_strides.size())
        THROW_IE_EXCEPTION << genLayer.getType() << " node " << genLayer.getName() << " contains incorrect parameters!";

    genLayer.getParameters()["kernel"] = l_kernel;
    genLayer.getParameters()["strides"] = l_strides;
    genLayer.getParameters()["pads_begin"] = l_paddingBegin;
    genLayer.getParameters()["pads_end"] = l_paddingEnd;
    return genLayer;
}

Builder::PoolingLayer& Builder::PoolingLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::PoolingLayer::getInputPort() const {
    return getLayer()->getInputPorts()[0];
}

Builder::PoolingLayer& Builder::PoolingLayer::setInputPort(const Port& port) {
    getLayer()->getInputPorts()[0] = port;
    return *this;
}

const Port& Builder::PoolingLayer::getOutputPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::PoolingLayer& Builder::PoolingLayer::setOutputPort(const Port& port) {
    getLayer()->getOutputPorts()[0] = port;
    return *this;
}

const std::vector<size_t> Builder::PoolingLayer::getKernel() const {
    return getLayer()->getParameters().at("kernel");
}
Builder::PoolingLayer& Builder::PoolingLayer::setKernel(const std::vector<size_t>& kernel) {
    getLayer()->getParameters()["kernel"] = kernel;
    return *this;
}

const std::vector<size_t> Builder::PoolingLayer::getStrides() const {
    return getLayer()->getParameters().at("strides");
}
Builder::PoolingLayer& Builder::PoolingLayer::setStrides(const std::vector<size_t>& strides) {
    getLayer()->getParameters()["strides"] = strides;
    return *this;
}

const std::vector<size_t> Builder::PoolingLayer::getPaddingsBegin() const {
    return getLayer()->getParameters().at("pads_begin");
}
Builder::PoolingLayer& Builder::PoolingLayer::setPaddingsBegin(const std::vector<size_t>& paddings) {
    getLayer()->getParameters()["pads_begin"] = paddings;
    return *this;
}

const std::vector<size_t> Builder::PoolingLayer::getPaddingsEnd() const {
    return getLayer()->getParameters().at("pads_end");
}
Builder::PoolingLayer& Builder::PoolingLayer::setPaddingsEnd(const std::vector<size_t>& paddings) {
    getLayer()->getParameters()["pads_end"] = paddings;
    return *this;
}

Builder::PoolingLayer::PoolingType Builder::PoolingLayer::getPoolingType() const {
    return type;
}
Builder::PoolingLayer& Builder::PoolingLayer::setPoolingType(Builder::PoolingLayer::PoolingType type) {
    std::string typeStr;
    switch (type) {
    case MAX:
        typeStr = "max";
        break;
    case AVG:
        typeStr = "avg";
        break;
    }
    getLayer()->getParameters()["pool-method"] = typeStr;
    this->type = type;
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
    getLayer()->getParameters()["rounding_type"] = typeStr;
    return *this;
}

bool Builder::PoolingLayer::getExcludePad() const {
    return getLayer()->getParameters().at("exclude-pad");
}

Builder::PoolingLayer& Builder::PoolingLayer::setExcludePad(bool exclude) {
    getLayer()->getParameters()["exclude-pad"] = exclude;
    return *this;
}

REG_VALIDATOR_FOR(Pooling, [](const Builder::Layer::CPtr& layer, bool partial) {
    // WA for old IRs
    if (layer->getParameters().find("kernel") == layer->getParameters().end() &&
        layer->getParameters().find("kernel-x") != layer->getParameters().end() &&
        layer->getParameters().find("kernel-y") != layer->getParameters().end())
        return;

    Builder::PoolingLayer poolBuilder(layer);
    std::vector<size_t> l_kernel = poolBuilder.getKernel();
    std::vector<size_t> l_paddingBegin = poolBuilder.getPaddingsBegin();
    std::vector<size_t> l_paddingEnd = poolBuilder.getPaddingsEnd();
    std::vector<size_t> l_strides = poolBuilder.getStrides();

    if (l_paddingBegin.empty() && !l_kernel.empty()) l_paddingBegin.resize(l_kernel.size(), 0);
    if (l_paddingEnd.empty() && !l_kernel.empty()) l_paddingEnd.resize(l_kernel.size(), 0);
    if (l_strides.empty() && !l_kernel.empty()) l_strides.resize(l_kernel.size(), 1);

    if (l_kernel.empty() || l_kernel.size() != l_paddingBegin.size() || l_kernel.size() != l_paddingEnd.size() ||
        l_kernel.size() != l_strides.size())
        THROW_IE_EXCEPTION << layer->getType() << " node " << layer->getName() << " contains incorrect parameters!";
});

REG_CONVERTER_FOR(Pooling, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    if (cnnLayer->params.find("kernel") == cnnLayer->params.end() &&
        cnnLayer->params.find("kernel-x") != cnnLayer->params.end() &&
        cnnLayer->params.find("kernel-y") != cnnLayer->params.end())
        return;
    std::vector<unsigned int> tmp = cnnLayer->GetParamAsUInts("kernel");
    layer.getParameters()["kernel"] = std::vector<size_t>(tmp.size());
    for (size_t i = 0; i < tmp.size(); ++i) {
        layer.getParameters()["kernel"].as<std::vector<size_t>>()[i] = static_cast<size_t>(tmp[i]);
    }

    tmp = cnnLayer->GetParamAsUInts("strides");
    layer.getParameters()["strides"] = std::vector<size_t>(tmp.size());
    for (size_t i = 0; i < tmp.size(); ++i) {
        layer.getParameters()["strides"].as<std::vector<size_t>>()[i] = static_cast<size_t>(tmp[i]);
    }

    tmp = cnnLayer->GetParamAsUInts("pads_begin");
    layer.getParameters()["pads_begin"] = std::vector<size_t>(tmp.size());
    for (size_t i = 0; i < tmp.size(); ++i) {
        layer.getParameters()["pads_begin"].as<std::vector<size_t>>()[i] = static_cast<size_t>(tmp[i]);
    }

    tmp = cnnLayer->GetParamAsUInts("pads_end");
    layer.getParameters()["pads_end"] = std::vector<size_t>(tmp.size());
    for (size_t i = 0; i < tmp.size(); ++i) {
        layer.getParameters()["pads_end"].as<std::vector<size_t>>()[i] = static_cast<size_t>(tmp[i]);
    }

    layer.getParameters()["exclude-pad"] = cnnLayer->GetParamAsBool("exclude-pad", false);
    layer.getParameters()["rounding_type"] = cnnLayer->GetParamAsString("rounding_type", "ceil");
    layer.getParameters()["pool-method"] = cnnLayer->GetParamAsString("pool-method", "max");
});
