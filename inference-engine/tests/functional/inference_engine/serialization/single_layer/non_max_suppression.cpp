// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/non_max_suppression.hpp"

using namespace ngraph;
using namespace LayerTestsDefinitions;

namespace {
    TEST_P(NmsLayerTest, Serialize) {
        Serialize();
    }

    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::FP16
    };

    /* ============= NO MAX SUPPRESSION ============= */

    const std::vector<InputShapeParams> inShapeParams = {
        InputShapeParams{3, 100, 5},
        InputShapeParams{1, 10, 50},
        InputShapeParams{2, 50, 50}
    };

    const std::vector<int32_t> maxOutBoxPerClass = {5, 20};
    const std::vector<float> threshold = {0.3f, 0.7f};
    const std::vector<float> sigmaThreshold = {0.0f, 0.5f};
    const std::vector<ngraph::op::v5::NonMaxSuppression::BoxEncodingType> encodType = {op::v5::NonMaxSuppression::BoxEncodingType::CENTER,
                                                                                    op::v5::NonMaxSuppression::BoxEncodingType::CORNER};
    const std::vector<bool> sortResDesc = {true, false};
    const std::vector<element::Type> outType = {element::i32, element::i64};

    const auto inPrecisions = ::testing::Combine(
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(InferenceEngine::Precision::I32),
        ::testing::Values(InferenceEngine::Precision::FP32));

    const auto nmsParams = ::testing::Combine(
            ::testing::ValuesIn(inShapeParams),
            inPrecisions,
            ::testing::ValuesIn(maxOutBoxPerClass),
            ::testing::ValuesIn(threshold), // IOU threshold
            ::testing::ValuesIn(threshold), // Score threshold
            ::testing::ValuesIn(sigmaThreshold),
            ::testing::ValuesIn(encodType),
            ::testing::ValuesIn(sortResDesc),
            ::testing::ValuesIn(outType),
            ::testing::Values(CommonTestUtils::DEVICE_CPU));

    INSTANTIATE_TEST_SUITE_P(smoke_NmsLayerTest, NmsLayerTest, nmsParams, NmsLayerTest::getTestCaseName);
}  // namespace

