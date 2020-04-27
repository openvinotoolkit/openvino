// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"

std::string ConvolutionGroupedTestModel::getName() const {
    return "ConvolutionGroupedTestModel";
}

void ConvolutionGroupedTestModel::initInput(Blob::Ptr input) const {
    fillDataWithInitValue(input, -1.f);
}

size_t ConvolutionGroupedTestModel::getGroupsCount(SingleLayerTransformationsTestParams& p) const {
    const size_t channelsPerGroup = 8ul;
    const size_t inputChannelsCount = p.inputDimensions[0][1];
    if ((inputChannelsCount % channelsPerGroup) != 0ul) {
        THROW_IE_EXCEPTION << "not possible to divide " << inputChannelsCount << " channels to groups";
    }

    return inputChannelsCount / channelsPerGroup;
}

bool ConvolutionGroupedTestModel::areScalesOnActivationsDifferent() const {
    return false;
}
