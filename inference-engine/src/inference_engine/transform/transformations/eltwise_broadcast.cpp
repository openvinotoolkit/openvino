// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_broadcast.hpp"
#include "builders/ie_network_builder.hpp"
#include "builders/ie_reshape_layer.hpp"
#include "builders/ie_tile_layer.hpp"
#include "debug.h"
#include <string>
#include <vector>
#include <iostream>

namespace InferenceEngine {
namespace Transform {

TransformationEltwiseBroadcast::TransformationEltwiseBroadcast() {
    this->setName("ie.transform.eltwise_broadcast");
}

void insertTileOverDimension(Transform::Network& network, Transform::Port& inputPort, size_t axis, size_t tile) {
    auto tileLayerBuilder = Builder::TileLayer("Tile" + std::to_string(axis) + "_" + std::to_string(tile)).setAxis(axis).setTiles(tile);
    auto tileLayer = network.addLayer(tileLayerBuilder);
    inputPort.getConnection().setDestination(tileLayer.getInPort());
    tileLayer.getOutPort().connect(inputPort);
}

void TransformationEltwiseBroadcast::execute(Network& network) {
    for (auto layer : network.getBuilderNetwork()) {
        if (layer->getType() == "Eltwise") {
            auto eltwiseLayer = network.getLayer(layer->getName());
            auto outShape = eltwiseLayer.getOutPort(0).shape();
            for (auto& eltwiseInPort : eltwiseLayer.getInPorts()) {
                auto inShape = eltwiseInPort.shape();
                // if shape lengths are not equal then insert Reshape with shape prepended with ones
                if (inShape.size() < outShape.size()) {
                    std::vector<int> reshapeDims(inShape.begin(), inShape.end());
                    reshapeDims.insert(reshapeDims.begin(), outShape.size() - inShape.size(), 1);
                    auto reshapeLayerBuilder = Builder::ReshapeLayer(eltwiseInPort.getLayer().getName() + "/Reshape").setDims(reshapeDims);
                    auto reshapeLayer = network.addLayer(reshapeLayerBuilder);
                    eltwiseInPort.getConnection().setDestination(reshapeLayer.getInPort());
                    reshapeLayer.getOutPort().connect(eltwiseInPort);
                    SizeVector newOutShape(reshapeDims.size());
                    // update shape of the Port
                    for (size_t ind = 0; ind < reshapeDims.size(); ++ind)
                        newOutShape[ind] = reshapeDims[ind];
                    eltwiseInPort.getData()->setShape(newOutShape);
                    inShape = newOutShape;
                }
                for (size_t axis = 0; axis < inShape.size(); ++axis) {
                    if (inShape[axis] != outShape[axis]) {
                        if (inShape[axis] != 1) {
                            THROW_IE_EXCEPTION << "Layer " << layer->getName()
                                               << " input has invalid shape "
                                               << details::dumpVec(inShape)
                                               << " which can not be broadcasted to output shape "
                                               << details::dumpVec(outShape);
                        }
                        insertTileOverDimension(network, eltwiseInPort, axis, outShape[axis]);
                    }
                }
            }
        }
    }
}

}  // namespace Transform
}  // namespace InferenceEngine
