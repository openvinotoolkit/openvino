// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/transpose.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
};

/**
 * 4D permute tests
 */
const std::vector<std::vector<size_t>> inputShapes = {
        std::vector<size_t>{1, 3, 100, 100},
        // use permute_8x8_4x4 kernel
        std::vector<size_t>{2, 8, 64, 64},
        std::vector<size_t>{2, 5, 64, 64},
        std::vector<size_t>{2, 8, 64, 5},
        std::vector<size_t>{2, 5, 64, 5},
};

const std::vector<std::vector<size_t>> inputOrder = {
        // use permute_ref
        std::vector<size_t>{0, 3, 2, 1},
        std::vector<size_t>{},
        // use permute_8x8_4x4 kernel
        std::vector<size_t>{0, 2, 3, 1},
};

const auto params = testing::Combine(
        testing::ValuesIn(inputOrder),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(inputShapes),
        testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Transpose,
        TransposeLayerTest,
        params,
        TransposeLayerTest::getTestCaseName
);

/**
 * 5D permute tests
 */
const std::vector<std::vector<size_t>> inputShapes5D = {
        std::vector<size_t>{2, 3, 4, 12, 64},
        std::vector<size_t>{2, 5, 11, 32, 32},
        std::vector<size_t>{2, 8, 64, 32, 5},
        std::vector<size_t>{2, 5, 64, 32, 5},
};

const std::vector<std::vector<size_t>> inputOrder5D = {
        // use permute_ref
        std::vector<size_t>{0, 3, 4, 2, 1},
        std::vector<size_t>{},
        // use permute_8x8_4x4 kernel
        std::vector<size_t>{0, 2, 3, 4, 1},
};

const auto params5D = testing::Combine(
        testing::ValuesIn(inputOrder5D),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(inputShapes5D),
        testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Transpose_5D,
        TransposeLayerTest,
        params5D,
        TransposeLayerTest::getTestCaseName
);

/**
 * 6D permute tests
 */
const std::vector<std::vector<size_t>> inputShapes6D = {
        std::vector<size_t>{2, 8, 5, 13, 11, 16},
        std::vector<size_t>{2, 11, 6, 2, 15, 10},
        std::vector<size_t>{2, 13, 1, 3, 14, 32},
        std::vector<size_t>{2, 14, 3, 4, 4, 22},
};

const std::vector<std::vector<size_t>> inputOrder6D = {
        // use permute_ref
        std::vector<size_t>{0, 4, 3, 5, 2, 1},
        std::vector<size_t>{},
        // use permute_8x8_4x4 kernel
        std::vector<size_t>{0, 2, 3, 4, 5, 1},
};

const auto params6D = testing::Combine(
        testing::ValuesIn(inputOrder6D),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(inputShapes6D),
        testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Transpose_6D,
        TransposeLayerTest,
        params6D,
        TransposeLayerTest::getTestCaseName
);


}  // namespace
