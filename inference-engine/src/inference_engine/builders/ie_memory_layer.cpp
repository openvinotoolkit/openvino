// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_memory_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::MemoryLayer::MemoryLayer(const std::string& name): LayerDecorator("Memory", name) {
    setSize(2);
}

Builder::MemoryLayer::MemoryLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("Memory");
}

Builder::MemoryLayer::MemoryLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("Memory");
}

Builder::MemoryLayer& Builder::MemoryLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::MemoryLayer::getInputPort() const {
    if (getLayer()->getInputPorts().empty()) {
        THROW_IE_EXCEPTION << "No inputs ports for layer: " << getLayer()->getName();
    }
    return getLayer()->getInputPorts()[0];
}

Builder::MemoryLayer& Builder::MemoryLayer::setInputPort(const Port &port) {
    getLayer()->getInputPorts().resize(1);
    getLayer()->getInputPorts()[0] = port;
    setIndex(0);
    return *this;
}

const Port& Builder::MemoryLayer::getOutputPort() const {
    if (getLayer()->getOutputPorts().empty()) {
        THROW_IE_EXCEPTION << "No output ports for layer: " << getLayer()->getName();
    }
    return getLayer()->getOutputPorts()[0];
}

Builder::MemoryLayer& Builder::MemoryLayer::setOutputPort(const Port &port) {
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getOutputPorts()[0] = port;
    setIndex(1);
    return *this;
}

const std::string Builder::MemoryLayer::getId() const {
    return getLayer()->getParameters().at("id");
}
Builder::MemoryLayer& Builder::MemoryLayer::setId(const std::string& id) {
    getLayer()->getParameters()["id"] = id;
    return *this;
}
size_t Builder::MemoryLayer::getIndex() const {
    return getLayer()->getParameters().at("index");
}
Builder::MemoryLayer& Builder::MemoryLayer::setIndex(size_t index) {
    if (index > 1)
        THROW_IE_EXCEPTION << "Index supports only 0 and 1 values.";
    getLayer()->getParameters()["index"] = index;
    return *this;
}
size_t Builder::MemoryLayer::getSize() const {
    return getLayer()->getParameters().at("size");
}
Builder::MemoryLayer& Builder::MemoryLayer::setSize(size_t size) {
    if (size != 2)
        THROW_IE_EXCEPTION << "Only size equal 2 is supported.";
    getLayer()->getParameters()["size"] = size;
    return *this;
}
REG_VALIDATOR_FOR(Memory, [](const InferenceEngine::Builder::Layer::CPtr& layer, bool partial) {
});

REG_CONVERTER_FOR(Memory, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    layer.getParameters()["id"] = cnnLayer->GetParamAsString("id", 0);
    layer.getParameters()["index"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("index", 0));
    layer.getParameters()["size"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("size", 0));
});

