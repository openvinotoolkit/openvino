// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/multiclass_nms.hpp"

using namespace ngraph;
using namespace ov::test::subgraph;

namespace {
TEST_P(MulticlassNmsLayerTest, Serialize) {
    serialize();
}

/* input format #1 with 2 inputs: bboxes N, M, 4, scores N, C, M */
const std::vector<std::vector<ov::test::InputShape>> shapeParams1 = {
    // num_batches, num_boxes, 4
    {{{ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic(), 4},
        {{1, 10, 4}, {2, 100, 4}}},
    // num_batches, num_classes, num_boxes
     {{ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic()},
        {{1, 3, 10}, {2, 5, 100}}}},
    {{{ngraph::Dimension(1, 10), ngraph::Dimension(1, 100), 4},
        {{1, 10, 4}, {2, 100, 4}}},
    {{{ngraph::Dimension(1, 10), ngraph::Dimension(1, 100), ngraph::Dimension(1, 100)}},
        {{1, 3, 10}, {2, 5, 100}}}}
};

/* input format #2 with 3 inputs: bboxes C, M, 4, scores C, M, roisnum N */
const std::vector<std::vector<ov::test::InputShape>> shapeParams2 = {
    /*0*/
    // bboxes
    {{{ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic(), 4},
        {{1, 10, 4}, {2, 100, 4}}},
    // scores
     {{ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic()},
        {{1, 10}, {2, 100}}},
    // roisnum
    {{ngraph::Dimension::dynamic()},
        {{1}, {10}}}},
    /*1*/
    {{{ngraph::Dimension(1, 10), ngraph::Dimension(1, 100), 4},
        {{1, 10, 4}, {2, 100, 4}}},
    {{{ngraph::Dimension(1, 10), ngraph::Dimension(1, 100)}},
        {{1, 10}, {2, 100}}},
    {{ngraph::Dimension::dynamic()},
        {{1}, {10}}}},
    /*2*/
    {{{ngraph::Dimension(3), ngraph::Dimension(2), 4},
        {{3, 2, 4}}},
    {{{ngraph::Dimension(3), ngraph::Dimension(2)}},
        {{3, 2}}},
    {{ngraph::Dimension::dynamic()},
        {{1}, {2}}}}
};

const std::vector<int32_t> nmsTopK = {-1, 20};
const std::vector<float> iouThreshold = {0.7f};
const std::vector<float> scoreThreshold = {0.7f};
const std::vector<int32_t> backgroundClass = {-1, 0};
const std::vector<int32_t> keepTopK = {-1, 30};
const std::vector<element::Type> outType = {element::i32, element::i64};

const std::vector<ov::op::util::MulticlassNmsBase::SortResultType> sortResultType = {
    op::v8::MulticlassNms::SortResultType::SCORE,
    op::v8::MulticlassNms::SortResultType::CLASSID,
    op::v8::MulticlassNms::SortResultType::NONE};
const std::vector<bool> sortResDesc = {true, false};
const std::vector<float> nmsEta = {0.6f, 1.0f};
const std::vector<bool> normalized = {true, false};

const auto nmsParams1_smoke = ::testing::Combine(
    ::testing::ValuesIn(shapeParams1),
    ::testing::Combine(::testing::Values(ov::element::f32),
                       ::testing::Values(ov::element::i32),
                       ::testing::Values(ov::element::f32)),
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

INSTANTIATE_TEST_SUITE_P(smoke_MulticlassNmsLayerTest1,
                        MulticlassNmsLayerTest,
                        nmsParams1_smoke,
                        MulticlassNmsLayerTest::getTestCaseName);


const auto nmsParams2_smoke = ::testing::Combine(
    ::testing::ValuesIn(shapeParams2),
    ::testing::Combine(::testing::Values(ov::element::f32),
                       ::testing::Values(ov::element::i32),
                       ::testing::Values(ov::element::f32)),
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

INSTANTIATE_TEST_SUITE_P(smoke_MulticlassNmsLayerTest2,
                        MulticlassNmsLayerTest,
                        nmsParams2_smoke,
                        MulticlassNmsLayerTest::getTestCaseName);
}  // namespace
