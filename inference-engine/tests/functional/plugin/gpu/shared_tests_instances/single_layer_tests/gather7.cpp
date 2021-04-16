// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/gather7.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisionsFP16 = {
        InferenceEngine::Precision::FP16,
};

const std::vector<std::vector<int>> indices = {
        std::vector<int>{0, 3, 2, 1},
};
const std::vector<std::vector<size_t>> indicesShapes2 = {
        std::vector<size_t>{2, 2}
};

const std::vector<int> batch_dim0 = {1};

const std::vector<std::vector<size_t>> inputShapesAxes4 = {
        std::vector<size_t>{2, 6, 7, 8, 9},
};

const std::vector<int> axes4 = {4};

const auto GatherAxes4 = testing::Combine(
        testing::ValuesIn(indices),
        testing::ValuesIn(indicesShapes2),
        testing::ValuesIn(axes4),
        testing::ValuesIn(inputShapesAxes4),
        testing::ValuesIn(netPrecisionsFP16),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_GPU)
);
const auto Gather7BatchDim4 = testing::Combine(
        GatherAxes4,
        testing::ValuesIn(batch_dim0)
);

INSTANTIATE_TEST_CASE_P(
        smoke_Gather7Axes4,
        Gather7LayerTest,
        Gather7BatchDim4,
        Gather7LayerTest::getTestCaseName
);

}  // namespace
