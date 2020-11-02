// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"

std::string ConvolutionDepthwiseTestModel::getName() const {
    return "ConvolutionDepthwiseTestModel";
}

size_t ConvolutionDepthwiseTestModel::getGroupsCount(SingleLayerTransformationsTestParams& p) const {
    return p.inputDimensions[0][1];
}

bool ConvolutionDepthwiseTestModel::areScalesOnActivationsDifferent() const {
    return true;
}
