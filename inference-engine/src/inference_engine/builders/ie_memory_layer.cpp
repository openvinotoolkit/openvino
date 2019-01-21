// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_memory_layer.hpp>
#include <details/caseless.hpp>

#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::MemoryLayer::MemoryLayer(const std::string& name): LayerFragment("Memory", name) {
    getLayer().getOutputPorts().resize(1);
    getLayer().getInputPorts().resize(1);
}

Builder::MemoryLayer::MemoryLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "Memory"))
        THROW_IE_EXCEPTION << "Cannot create MemoryLayer decorator for layer " << getLayer().getType();
}

Builder::MemoryLayer& Builder::MemoryLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}

const Port& Builder::MemoryLayer::getInputPort() const {
    return getLayer().getInputPorts()[0];
}

Builder::MemoryLayer& Builder::MemoryLayer::setInputPort(const Port &port) {
    getLayer().getInputPorts()[0] = port;
    return *this;
}

const Port& Builder::MemoryLayer::getOutputPort() const {
    return getLayer().getOutputPorts()[0];
}

Builder::MemoryLayer& Builder::MemoryLayer::setOutputPort(const Port &port) {
    getLayer().getOutputPorts()[0] = port;
    return *this;
}

const std::string Builder::MemoryLayer::getId() const {
    return getLayer().getParameters()["id"];
}
Builder::MemoryLayer& Builder::MemoryLayer::setId(const std::string& id) {
    getLayer().getParameters()["id"] = id;
    return *this;
}
size_t Builder::MemoryLayer::getIndex() const {
    return getLayer().getParameters()["index"].asUInt();
}
Builder::MemoryLayer& Builder::MemoryLayer::setIndex(size_t index) {
    if (index > 1)
        THROW_IE_EXCEPTION << "Index supports only 0 and 1 values.";
    getLayer().getParameters()["index"] = index;
    return *this;
}
size_t Builder::MemoryLayer::getSize() const {
    return getLayer().getParameters()["size"].asUInt(2);
}
Builder::MemoryLayer& Builder::MemoryLayer::setSize(size_t size) {
    if (size != 2)
        THROW_IE_EXCEPTION << "Only size equal 2 is supported.";
    getLayer().getParameters()["size"] = size;
    return *this;
}
