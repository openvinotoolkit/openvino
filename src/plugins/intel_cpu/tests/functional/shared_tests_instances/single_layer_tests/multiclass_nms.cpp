// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/multiclass_nms.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace ov::test::subgraph;
using namespace InferenceEngine;
using namespace ngraph;

/* input format #1 with 2 inputs: bboxes N, M, 4, scores N, C, M */
const std::vector<std::vector<ov::Shape>> inStaticShapeParams1 = {
    {{3, 100, 4}, {3,   1, 100}},
    {{1, 10,  4}, {1, 100, 10 }}
};

const std::vector<std::vector<ov::test::InputShape>> inDynamicShapeParams1 = {
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
const std::vector<std::vector<ov::Shape>> inStaticShapeParams2 = {
    {{1, 10, 4}, {1, 10}, {1}},
    {{1, 10, 4}, {1, 10}, {10}},
    {{2, 100, 4}, {2, 100}, {1}},
    {{2, 100, 4}, {2, 100}, {10}}
};

const std::vector<std::vector<ov::test::InputShape>> inDynamicShapeParams2 = {
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
    {{{ngraph::Dimension(2), ngraph::Dimension(100), 4},
        {{2, 100, 4}}},
    {{{ngraph::Dimension(2), ngraph::Dimension(100)}},
        {{2, 100}}},
    {{ngraph::Dimension::dynamic()},
        {{1}, {10}}}},
    /*3*/
    {{{ngraph::Dimension(3), ngraph::Dimension(2), 4},
        {{3, 2, 4}}},
    {{{ngraph::Dimension(3), ngraph::Dimension(2)}},
        {{3, 2}}},
    {{ngraph::Dimension::dynamic()},
        {{1}, {10}}}}     // more images than num_boxes
};

const std::vector<int32_t> nmsTopK = {-1, 20};
const std::vector<float> iouThreshold = {0.7f};
const std::vector<float> scoreThreshold = {0.7f};
const std::vector<int32_t> backgroundClass = {-1, 1};
const std::vector<int32_t> keepTopK = {-1, 30};
const std::vector<element::Type> outType = {element::i32, element::i64};

const std::vector<ov::op::util::MulticlassNmsBase::SortResultType> sortResultType = {
    ov::op::util::MulticlassNmsBase::SortResultType::SCORE,
    ov::op::util::MulticlassNmsBase::SortResultType::CLASSID,
    ov::op::util::MulticlassNmsBase::SortResultType::NONE};
const std::vector<bool> sortResDesc = {true, false};
const std::vector<float> nmsEta = {0.6f, 1.0f};
const std::vector<bool> normalized = {true, false};

const std::vector<bool> outStaticShape = {false};   // only be false for cpu plugin with ov2.0.

const auto nmsParamsStatic_smoke1 = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inStaticShapeParams1)),
    ::testing::Combine(::testing::Values(ov::element::f32),
                       ::testing::Values(ov::element::i32),
                       ::testing::Values(ov::element::i32),
                       ::testing::Values(ov::element::f32)),
    ::testing::ValuesIn(nmsTopK),
    ::testing::Combine(::testing::ValuesIn(iouThreshold), ::testing::ValuesIn(scoreThreshold), ::testing::ValuesIn(nmsEta)),
    ::testing::ValuesIn(backgroundClass),
    ::testing::ValuesIn(keepTopK),
    ::testing::ValuesIn(outType),
    ::testing::ValuesIn(sortResultType),
    ::testing::Combine(::testing::ValuesIn(sortResDesc), ::testing::ValuesIn(normalized)),
    ::testing::ValuesIn(outStaticShape),
    ::testing::Values(ov::test::utils::DEVICE_CPU));

const auto nmsParamsDynamic_smoke1 = ::testing::Combine(
    ::testing::ValuesIn(inDynamicShapeParams1),
    ::testing::Combine(::testing::Values(ov::element::f32),
                       ::testing::Values(ov::element::i32),
                       ::testing::Values(ov::element::i32),
                       ::testing::Values(ov::element::f32)),
    ::testing::ValuesIn(nmsTopK),
    ::testing::Combine(::testing::ValuesIn(iouThreshold), ::testing::ValuesIn(scoreThreshold), ::testing::ValuesIn(nmsEta)),
    ::testing::ValuesIn(backgroundClass),
    ::testing::ValuesIn(keepTopK),
    ::testing::ValuesIn(outType),
    ::testing::ValuesIn(sortResultType),
    ::testing::Combine(::testing::ValuesIn(sortResDesc), ::testing::ValuesIn(normalized)),
    ::testing::ValuesIn(outStaticShape),
    ::testing::Values(ov::test::utils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(smoke_MulticlassNmsLayerTest_static1, MulticlassNmsLayerTest, nmsParamsStatic_smoke1, MulticlassNmsLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MulticlassNmsLayerTest_dynamic1, MulticlassNmsLayerTest, nmsParamsDynamic_smoke1, MulticlassNmsLayerTest::getTestCaseName);

const auto nmsParamsStatic_smoke2 = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inStaticShapeParams2)),
    ::testing::Combine(::testing::Values(ov::element::f32),
                       ::testing::Values(ov::element::i32),
                       ::testing::Values(ov::element::i32),
                       ::testing::Values(ov::element::f32)),
    ::testing::ValuesIn(nmsTopK),
    ::testing::Combine(::testing::ValuesIn(iouThreshold), ::testing::ValuesIn(scoreThreshold), ::testing::ValuesIn(nmsEta)),
    ::testing::ValuesIn(backgroundClass),
    ::testing::ValuesIn(keepTopK),
    ::testing::ValuesIn(outType),
    ::testing::ValuesIn(sortResultType),
    ::testing::Combine(::testing::ValuesIn(sortResDesc), ::testing::ValuesIn(normalized)),
    ::testing::ValuesIn(outStaticShape),
    ::testing::Values(ov::test::utils::DEVICE_CPU));

const auto nmsParamsDynamic_smoke2 = ::testing::Combine(
    ::testing::ValuesIn(inDynamicShapeParams2),
    ::testing::Combine(::testing::Values(ov::element::f32),
                       ::testing::Values(ov::element::i32),
                       ::testing::Values(ov::element::i32),
                       ::testing::Values(ov::element::f32)),
    ::testing::ValuesIn(nmsTopK),
    ::testing::Combine(::testing::ValuesIn(iouThreshold), ::testing::ValuesIn(scoreThreshold), ::testing::ValuesIn(nmsEta)),
    ::testing::ValuesIn(backgroundClass),
    ::testing::ValuesIn(keepTopK),
    ::testing::ValuesIn(outType),
    ::testing::ValuesIn(sortResultType),
    ::testing::Combine(::testing::ValuesIn(sortResDesc), ::testing::ValuesIn(normalized)),
    ::testing::ValuesIn(outStaticShape),
    ::testing::Values(ov::test::utils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(smoke_MulticlassNmsLayerTest_static2, MulticlassNmsLayerTest, nmsParamsStatic_smoke2, MulticlassNmsLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MulticlassNmsLayerTest_dynamic2, MulticlassNmsLayerTest, nmsParamsDynamic_smoke2, MulticlassNmsLayerTest::getTestCaseName);
