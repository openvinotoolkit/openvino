// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/pooling.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace ngraph::helpers;
using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<std::vector<size_t >> kernels = {{3, 3},
                                                          {3, 5}};
const std::vector<std::vector<size_t >> strides = {{1, 1},
                                                          {1, 2}};
const std::vector<std::vector<size_t >> padBegins = {{0, 0},
                                                            {0, 2}};
const std::vector<std::vector<size_t >> padEnds = {{0, 0},
                                                          {0, 2}};
const std::vector<ngraph::op::RoundingType> roundingTypes = {ngraph::op::RoundingType::CEIL,
                                                             ngraph::op::RoundingType::FLOOR};

////* ========== Max Polling ========== */
/* +========== Explicit Pad Floor Rounding ========== */
const auto maxPool_ExplicitPad_FloorRounding_Params = ::testing::Combine(
        ::testing::Values(PoolingTypes::MAX),
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds),
        ::testing::Values(ngraph::op::RoundingType::FLOOR),
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::Values(false)  // placeholder value - exclude pad not applicable for max pooling
);

INSTANTIATE_TEST_CASE_P(smoke_MaxPool_ExplicitPad_FloorRpunding, PoolingLayerTest,
                        ::testing::Combine(
                                maxPool_ExplicitPad_FloorRounding_Params,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(std::vector<size_t >({1, 3, 50, 50})),
                                ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        PoolingLayerTest::getTestCaseName);

/* ========== Explicit Pad Ceil Rounding ========== */
const auto maxPool_ExplicitPad_CeilRounding_Params = ::testing::Combine(
        ::testing::Values(PoolingTypes::MAX),
        ::testing::ValuesIn(kernels),
        // TODO: Non 1 strides fails in ngraph reference implementation with error "The end corner is out of bounds at axis 3" thrown in the test body.
        ::testing::Values(std::vector<size_t>({1, 1})),
        ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds),
        ::testing::Values(ngraph::op::RoundingType::CEIL),
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::Values(false)  // placeholder value - exclude pad not applicable for max pooling
);

INSTANTIATE_TEST_CASE_P(smoke_MaxPool_ExplicitPad_CeilRpunding, PoolingLayerTest,
                        ::testing::Combine(
                                maxPool_ExplicitPad_CeilRounding_Params,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(std::vector<size_t >({1, 3, 50, 50})),
                                ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        PoolingLayerTest::getTestCaseName);


////* ========== Avg Pooling ========== */
/* +========== Explicit Pad Ceil Rounding ========== */
const auto avgPoolExplicitPadCeilRoundingParams = ::testing::Combine(
        ::testing::Values(PoolingTypes::AVG),
        ::testing::ValuesIn(kernels),
        // TODO: Non 1 strides fails in ngraph reference implementation with error "The end corner is out of bounds at axis 3" thrown in the test body.
        ::testing::Values(std::vector<size_t>({1, 1})),
        ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds),
        ::testing::Values(ngraph::op::RoundingType::CEIL),
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::Values(true, false)
);

INSTANTIATE_TEST_CASE_P(smoke_AvgPool_ExplicitPad_CeilRounding, PoolingLayerTest,
                       ::testing::Combine(
                               avgPoolExplicitPadCeilRoundingParams,
                               ::testing::ValuesIn(netPrecisions),
                               ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                               ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                               ::testing::Values(InferenceEngine::Layout::ANY),
                               ::testing::Values(InferenceEngine::Layout::ANY),
                               ::testing::Values(std::vector<size_t >({1, 3, 30, 30})),
                               ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                       PoolingLayerTest::getTestCaseName);

/* +========== Explicit Pad Floor Rounding ========== */
const auto avgPoolExplicitPadFloorRoundingParams = ::testing::Combine(
        ::testing::Values(PoolingTypes::AVG),
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds),
        ::testing::Values(ngraph::op::RoundingType::FLOOR),
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::Values(true, false)
);


INSTANTIATE_TEST_CASE_P(smoke_AvgPool_ExplicitPad_FloorRounding, PoolingLayerTest,
                        ::testing::Combine(
                                avgPoolExplicitPadFloorRoundingParams,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(std::vector<size_t >({1, 3, 30, 30})),
                                ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        PoolingLayerTest::getTestCaseName);

////* ========== Avg and Max Polling Cases ========== */
/*    ========== Valid Pad Rounding Not Applicable ========== */
const auto allPools_ValidPad_Params = ::testing::Combine(
        ::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::Values(std::vector<size_t>({0, 0})),
        ::testing::ValuesIn(padEnds),
        ::testing::Values(ngraph::op::RoundingType::FLOOR),  // placeholder value - Rounding Type not applicable for Valid pad type
        // TODO: PadType::VALID seems not to ignore padBegins
        ::testing::Values(ngraph::op::PadType::VALID),
        ::testing::Values(false)  // placeholder value - exclude pad not applicable for max pooling
);

INSTANTIATE_TEST_CASE_P(smoke_MAX_and_AVGPool_ValidPad, PoolingLayerTest,
                        ::testing::Combine(
                                allPools_ValidPad_Params,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(std::vector<size_t >({1, 3, 50, 50})),
                                ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        PoolingLayerTest::getTestCaseName);
}  // namespace
