// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/multiclass_nms.hpp"

using namespace ngraph;
using namespace LayerTestsDefinitions;

namespace {
TEST_P(MulticlassNmsLayerTest, Serialize) {
    Serialize();
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16};

const std::vector<InputShapeParams> inStaticShapeParams = {
        // dynamic shape, {{batch, box, 4}, {batch, class, box}}
        {{}, {{{3, 100, 4}, {3,   5, 100}}}},
        {{{ngraph::Dimension::dynamic(), 100, 4}, {ngraph::Dimension::dynamic(), 5, 100}},
            {{{1, 100, 4}, {1, 5, 100}}, {{2, 100, 4}, {2, 5, 100}}, {{3, 100, 4}, {3, 5, 100}}}}
};

const std::vector<int32_t> nmsTopK = {-1, 20};
const std::vector<float> iouThreshold = {0.7f};
const std::vector<float> scoreThreshold = {0.7f};
const std::vector<int32_t> backgroundClass = {-1, 0};
const std::vector<int32_t> keepTopK = {-1, 30};
const std::vector<element::Type> outType = {element::i32, element::i64};

const std::vector<op::v8::MulticlassNms::SortResultType> sortResultType = {
    op::v8::MulticlassNms::SortResultType::SCORE,
    op::v8::MulticlassNms::SortResultType::CLASSID,
    op::v8::MulticlassNms::SortResultType::NONE};
const std::vector<bool> sortResDesc = {true, false};
const std::vector<float> nmsEta = {0.6f, 1.0f};
const std::vector<bool> normalized = {true, false};

const auto nmsParams = ::testing::Combine(
    ::testing::ValuesIn(inStaticShapeParams),
    ::testing::Combine(::testing::Values(InferenceEngine::Precision::FP32),
                       ::testing::Values(InferenceEngine::Precision::I32),
                       ::testing::Values(InferenceEngine::Precision::FP32)),
    ::testing::ValuesIn(nmsTopK),
    ::testing::Combine(::testing::ValuesIn(iouThreshold),
                       ::testing::ValuesIn(scoreThreshold),
                       ::testing::ValuesIn(nmsEta)),
    ::testing::ValuesIn(backgroundClass),
    ::testing::ValuesIn(keepTopK),
    ::testing::ValuesIn(outType),
    ::testing::ValuesIn(sortResultType),
    ::testing::Combine(::testing::ValuesIn(sortResDesc),
                       ::testing::ValuesIn(normalized)),
    ::testing::Values(CommonTestUtils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(smoke_MulticlassNmsLayerTest,
                        MulticlassNmsLayerTest,
                        nmsParams,
                        MulticlassNmsLayerTest::getTestCaseName);
}  // namespace
