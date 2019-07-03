// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sub.hpp"
#include "builders/ie_network_builder.hpp"
#include "builders/ie_power_layer.hpp"
#include "builders/ie_eltwise_layer.hpp"
#include <vector>
#include <string>
#include <iostream>

namespace InferenceEngine {
namespace Transform {

TransformationSub::TransformationSub() {
    this->setName("ie.transform.sub");
}

void TransformationSub::execute(Network& network) {
    for (auto layer : network.getBuilderNetwork()) {
        if (layer->getType() == "Eltwise" && layer->getParameters()["operation"].as<std::string>() == "sub") {
            auto subLayer = network.getLayer(layer->getName());

            auto powerLayerBuilder = Builder::PowerLayer(subLayer.getName() + "/Power").setPower(1.0f).setScale(-1.0f).setShift(0.0f);
            auto powerLayer = network.addLayer(powerLayerBuilder);

            auto eltwiseLayerBuilder = Builder::EltwiseLayer(subLayer.getName() + "/Add").setEltwiseType(Builder::EltwiseLayer::EltwiseType::SUM);
            auto eltwiseLayer = network.addLayer(eltwiseLayerBuilder);

            // negate the second input to the sub layer
            subLayer.getInPort(1).getConnection().setDestination(powerLayer.getInPort());

            // connect new eltwise with sum with two inputs
            subLayer.getInPort(0).getConnection().setDestination(eltwiseLayer.getInPort(0));
            eltwiseLayer.getInPort(1).connect(powerLayer.getOutPort());

            // reconnect new eltwise with outputs of all eltwise with sub
            subLayer.getOutPort().getConnection().setSource(eltwiseLayer.getOutPort());

            network.removeLayer(subLayer);
        }
    }
}

}  // namespace Transform
}  // namespace InferenceEngine
