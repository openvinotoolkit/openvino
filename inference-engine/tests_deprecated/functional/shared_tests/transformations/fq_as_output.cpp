// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"

std::string FakeQuantizeAsOutputTest::getName() const {
    return "FakeQuantizeAsOutputTest";
}

bool FakeQuantizeAsOutputTest::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    network.addOutput("FakeQuantize12");

    LowPrecisionTransformer transformer(LowPrecisionTransformer::getAllTransformations(params));
    transformer.transform(network);

    const auto fq = network.getLayerByName("FakeQuantize12");
    if (fq == nullptr)
        THROW_IE_EXCEPTION << "Layer 'FakeQuantize12' should not be transformed";

    return true;
}

std::unordered_set<std::string> FakeQuantizeAsOutputTest::getNotTransformedLayers() const {
    return { "Convolution14" };
}
