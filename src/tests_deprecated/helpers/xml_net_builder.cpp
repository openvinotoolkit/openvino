// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include <algorithm>
#include <string>
#include "xml_net_builder.hpp"

namespace CommonTestUtils {

size_t IDManager::getNextLayerID() {
    return layerID++;
}

size_t IDManager::getNextPortID() {
    return portID++;
}

void IDManager::reset() {
    portID = layerID = 0;
}

LayerDesc::LayerDesc(std::string type, InOutShapes &shapes, IDManager &id_manager)
        : _type(std::move(type)) {
    _layerID = id_manager.getNextLayerID();
    auto inDims = shapes.inDims;
    auto outDims = shapes.outDims;
    for (const auto &inDim : inDims) {
        _inPortsID.emplace_back(id_manager.getNextPortID(), inDim);
    }
    for (const auto &outDim : outDims) {
        _outPortsID.emplace_back(id_manager.getNextPortID(), outDim);
    }
}

void LayerDesc::resetPortIDs() {
    _currentInPort = _currentOutPort = 0;
}

LayerDesc::LayerPortData LayerDesc::getNextInData() {
    if (_currentInPort == _inPortsID.size())
        IE_THROW() << "Failed to get next input port: reached the last one";
    return _inPortsID[_currentInPort++];
}

LayerDesc::LayerPortData LayerDesc::getNextOutData() {
    if (_currentOutPort == _outPortsID.size())
        IE_THROW() << "Failed to get next output port: reached the last one";
    return _outPortsID[_currentOutPort++];
}

size_t LayerDesc::getLayerID() const {
    return _layerID;
}

size_t LayerDesc::getInputsSize() const {
    return _inPortsID.size();
}

size_t LayerDesc::getOutputsSize() const {
    return _outPortsID.size();
}

std::string LayerDesc::getLayerName() const {
    return _type + std::to_string(getLayerID());
}


EdgesBuilder &EdgesBuilder::connect(size_t layer1, size_t layer2) {
    auto found1 = std::find_if(layersDesc.begin(), layersDesc.end(), [&layer1](const LayerDesc::Ptr &desc) {
        return desc->getLayerID() == layer1;
    });
    auto found2 = std::find_if(layersDesc.begin(), layersDesc.end(), [&layer2](const LayerDesc::Ptr &desc) {
        return desc->getLayerID() == layer2;
    });
    if (found1 == layersDesc.end() || found2 == layersDesc.end())
        IE_THROW() << "Failed to find layers with index: " << layer1 << " and " << layer2;

    nodeEdges.node("edge")
            .attr("from-layer", (*found1)->getLayerID())
            .attr("from-port", (*found1)->getNextOutData().portID)
            .attr("to-layer", (*found2)->getLayerID())
            .attr("to-port", (*found2)->getNextInData().portID).close();
    return *this;
}

std::string EdgesBuilder::finish() {
    auto &exp = nodeEdges.close();
    return exp;
}

}  // namespace CommonTestUtils