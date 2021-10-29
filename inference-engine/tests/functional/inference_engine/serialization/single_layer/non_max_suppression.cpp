// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/non_max_suppression.hpp"

using namespace ngraph;
using namespace ov::test::subgraph;

namespace {
    TEST_P(NmsLayerTest, Serialize) {
        serialize();
    }

    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::FP16
    };

    /* ============= NO MAX SUPPRESSION ============= */

    const std::vector<InputShapeParams> inShapeParams = {
        InputShapeParams{std::vector<ov::Dimension>{}, std::vector<TargetShapeParams>{{3, 100, 5}}},
        InputShapeParams{std::vector<ov::Dimension>{}, std::vector<TargetShapeParams>{{1, 10, 50}}},
        InputShapeParams{std::vector<ov::Dimension>{}, std::vector<TargetShapeParams>{{2, 50, 50}}}
    };

    const std::vector<int32_t> maxOutBoxPerClass = {5, 20};
    const std::vector<float> threshold = {0.3f, 0.7f};
    const std::vector<float> sigmaThreshold = {0.0f, 0.5f};
    const std::vector<ngraph::op::v5::NonMaxSuppression::BoxEncodingType> encodType = {op::v5::NonMaxSuppression::BoxEncodingType::CENTER,
                                                                                       op::v5::NonMaxSuppression::BoxEncodingType::CORNER};
    const std::vector<bool> sortResDesc = {true, false};
    const std::vector<element::Type> outType = {element::i32, element::i64};

    const auto inPrecisions = ::testing::Combine(
        ::testing::Values(ov::element::Type_t::f32),
        ::testing::Values(ov::element::Type_t::i32),
        ::testing::Values(ov::element::Type_t::f32));

    const auto nmsParams = ::testing::Combine(
            ::testing::ValuesIn(inShapeParams),
            inPrecisions,
            ::testing::ValuesIn(maxOutBoxPerClass),
            ::testing::Combine(::testing::ValuesIn(threshold),
                               ::testing::ValuesIn(threshold),
                               ::testing::ValuesIn(sigmaThreshold)),
            ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
            ::testing::ValuesIn(encodType),
            ::testing::ValuesIn(sortResDesc),
            ::testing::ValuesIn(outType),
            ::testing::Values(CommonTestUtils::DEVICE_CPU));

    INSTANTIATE_TEST_SUITE_P(smoke_NmsLayerTest, NmsLayerTest, nmsParams, NmsLayerTest::getTestCaseName);
}  // namespace

