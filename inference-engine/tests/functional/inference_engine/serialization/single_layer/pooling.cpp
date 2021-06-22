// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/pooling.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
    TEST_P(PoolingLayerTest, Serialize) {
        Serialize();
    }

    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::FP16
    };

    /* ============= POOLING ============= */
    const std::vector<std::vector<size_t >> kernels = {{3, 3},
                                                   {3, 5}};
    const std::vector<std::vector<size_t >> strides = {{1, 1},
                                                    {1, 2}};
    const std::vector<std::vector<size_t >> padBegins = {{0, 0},
                                                        {0, 2}};
    const std::vector<std::vector<size_t >> padEnds = {{0, 0},
                                                    {0, 2}};

    const std::vector<ngraph::op::RoundingType> roundingTypes = {
       ngraph::op::RoundingType::FLOOR,
       ngraph::op::RoundingType::CEIL
    };

    const std::vector<ngraph::op::PadType> padTypes = {
        ngraph::op::PadType::EXPLICIT,
        ngraph::op::PadType::SAME_UPPER,
        ngraph::op::PadType::VALID
    };

    const std::vector<size_t> inputShape = {511, 11, 13, 15};
    const std::vector<bool> excludePad = {true, false};

    /* ============= AVERAGE POOLING ============= */

    const auto avgExcludePadParams = ::testing::Combine(
            ::testing::Values(ngraph::helpers::PoolingTypes::AVG),
            ::testing::ValuesIn(kernels),
            ::testing::ValuesIn(strides),
            ::testing::ValuesIn(padBegins),
            ::testing::ValuesIn(padEnds),
            ::testing::ValuesIn(roundingTypes),
            ::testing::ValuesIn(padTypes),
            ::testing::Values(excludePad[0]));

    INSTANTIATE_TEST_SUITE_P(smoke_AvgPoolExcluePad, PoolingLayerTest,
            ::testing::Combine(
                avgExcludePadParams,
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(inputShape),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
            PoolingLayerTest::getTestCaseName);

    const auto avgPadParams = ::testing::Combine(
            ::testing::Values(ngraph::helpers::PoolingTypes::AVG),
            ::testing::ValuesIn(kernels),
            ::testing::ValuesIn(strides),
            ::testing::ValuesIn(padBegins),
            ::testing::ValuesIn(padEnds),
            ::testing::ValuesIn(roundingTypes),
            ::testing::ValuesIn(padTypes),
            ::testing::Values(excludePad[1]));

    INSTANTIATE_TEST_SUITE_P(smoke_AvgPool, PoolingLayerTest,
            ::testing::Combine(
                avgPadParams,
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(inputShape),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
            PoolingLayerTest::getTestCaseName);
}  // namespace

