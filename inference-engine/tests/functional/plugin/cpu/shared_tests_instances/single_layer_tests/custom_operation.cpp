// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/custom_operation.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<std::vector<size_t>> inputShapes = {
        {1, 3},
        {2, 5},
        {1, 3, 10},
        {1, 3, 1, 1},
        {2, 5, 4, 4},
};


const auto customOpParams = testing::Combine(
        testing::ValuesIn(netPrecisions),
        testing::ValuesIn(inputShapes),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(
        smoke_CustomOperation,
        CustomOpLayerTest,
        customOpParams,
        CustomOpLayerTest::getTestCaseName
);

}  // namespace

