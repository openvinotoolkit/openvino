// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_tile_layer.hpp>
#include <details/caseless.hpp>

#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::TileLayer::TileLayer(const std::string& name): LayerFragment("Tile", name) {
    getLayer().getOutputPorts().resize(1);
    getLayer().getInputPorts().resize(1);
}

Builder::TileLayer::TileLayer(Layer& genLayer): LayerFragment(genLayer) {
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "Tile"))
        THROW_IE_EXCEPTION << "Cannot create TileLayer decorator for layer " << getLayer().getType();
}

Builder::TileLayer& Builder::TileLayer::setName(const std::string& name) {
    getLayer().getName() = name;
    return *this;
}

const Port& Builder::TileLayer::getInputPort() const {
    return getLayer().getInputPorts()[0];
}

Builder::TileLayer& Builder::TileLayer::setInputPort(const Port &port) {
    getLayer().getInputPorts()[0] = port;
    return *this;
}

const Port& Builder::TileLayer::getOutputPort() const {
    return getLayer().getOutputPorts()[0];
}

Builder::TileLayer& Builder::TileLayer::setOutputPort(const Port &port) {
    getLayer().getOutputPorts()[0] = port;
    return *this;
}

size_t Builder::TileLayer::getTiles() const {
    return getLayer().getParameters()["tiles"].asUInt();
}

Builder::TileLayer& Builder::TileLayer::setTiles(size_t tiles) {
    getLayer().getParameters()["tiles"] = tiles;
    return *this;
}

size_t Builder::TileLayer::getAxis() const {
    return getLayer().getParameters()["axis"].asUInt();
}

Builder::TileLayer& Builder::TileLayer::setAxis(size_t axis) {
    getLayer().getParameters()["axis"] = axis;
    return *this;
}