// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_tile_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::TileLayer::TileLayer(const std::string& name): LayerDecorator("Tile", name) {
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getInputPorts().resize(1);
}

Builder::TileLayer::TileLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    checkType("Tile");
}

Builder::TileLayer::TileLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    checkType("Tile");
}

Builder::TileLayer& Builder::TileLayer::setName(const std::string& name) {
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::TileLayer::getInputPort() const {
    return getLayer()->getInputPorts()[0];
}

Builder::TileLayer& Builder::TileLayer::setInputPort(const Port &port) {
    getLayer()->getInputPorts()[0] = port;
    return *this;
}

const Port& Builder::TileLayer::getOutputPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::TileLayer& Builder::TileLayer::setOutputPort(const Port &port) {
    getLayer()->getOutputPorts()[0] = port;
    return *this;
}

size_t Builder::TileLayer::getTiles() const {
    return getLayer()->getParameters().at("tiles");
}

Builder::TileLayer& Builder::TileLayer::setTiles(size_t tiles) {
    getLayer()->getParameters()["tiles"] = tiles;
    return *this;
}

size_t Builder::TileLayer::getAxis() const {
    return getLayer()->getParameters().at("axis");
}

Builder::TileLayer& Builder::TileLayer::setAxis(size_t axis) {
    getLayer()->getParameters()["axis"] = axis;
    return *this;
}

REG_CONVERTER_FOR(Tile, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    layer.getParameters()["axis"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("axis"));
    layer.getParameters()["tiles"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("tiles"));
});