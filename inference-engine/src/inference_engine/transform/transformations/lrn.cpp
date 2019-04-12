// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lrn.hpp"
#include "builders/ie_network_builder.hpp"
#include "builders/ie_power_layer.hpp"
#include "builders/ie_eltwise_layer.hpp"
#include "builders/ie_norm_layer.hpp"
#include <iostream>
#include <cmath>

namespace InferenceEngine {
namespace Transform {

TransformationLRN::TransformationLRN() {
    this->setName("ie.transform.lrn");
}

void TransformationLRN::execute(Network& network) {
    for (auto layer : network.getBuilderNetwork()) {
        if (layer->getType() == "LRN") {
            auto lrnLayer = network.getLayer(layer->getName());
            float scale_value = 1.0f / std::pow(static_cast<float>(lrnLayer.getParameter("bias")),
                                                static_cast<float>(lrnLayer.getParameter("beta")));

            auto normLayerBuilder = Builder::NormLayer(lrnLayer.getName() + "/Norm").
                    setAlpha(static_cast<float>(lrnLayer.getParameter("alpha")) / static_cast<float>(lrnLayer.getParameter("bias"))).
                    setSize(static_cast<unsigned int>(lrnLayer.getParameter("size"))).
                    setBeta(static_cast<float>(lrnLayer.getParameter("beta"))).
                    setAcrossMaps(true);
            auto normLayer = network.addLayer(normLayerBuilder);

            auto mulLayerBuilder = Builder::EltwiseLayer(lrnLayer.getName() + "/Mul").setEltwiseType(
                    Builder::EltwiseLayer::EltwiseType::MUL);
            auto mulLayer = network.addLayer(mulLayerBuilder);

            auto tensorDesc = TensorDesc(Precision::FP32, SizeVector(4, 1), Layout::NCHW);
            auto blob = make_shared_blob<float>(tensorDesc);
            blob->allocate();
            float *buffer = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
            buffer[0] = scale_value;

            auto constLayerBuilder = Builder::ConstLayer(mulLayerBuilder.getName() + "/Const").setData(blob);
            auto constLayer = network.addLayer(constLayerBuilder);

            // re-connect input of LRN layer to input of Norm layer
            lrnLayer.getInPort().getConnection().setDestination(normLayer.getInPort());

            // multiple output of Norm with a constant
            mulLayer.getInPort(0).connect(normLayer.getOutPort());
            mulLayer.getInPort(1).connect(constLayer.getOutPort());

            // connect consumers of LRN with mul
            lrnLayer.getOutPort().getConnection().setSource(mulLayer.getOutPort());

            network.removeLayer(lrnLayer);
        }
    }
}

}  // namespace Transform
}  // namespace InferenceEngine
