// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/multiclass_nms.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace ov::test::subgraph;
using namespace InferenceEngine;
using namespace ngraph;

const std::vector<ShapeParams> inStaticShapeParams = {
    // dynamic shape, {{batch, box, 4}, {batch, class, box}}, out if static shape
    ShapeParams{{}, {{{3, 100, 4}, {3,   1, 100}}}, true},
    ShapeParams{{}, {{{1, 10,  4}, {1, 100, 10 }}}, false}
};

const std::vector<ShapeParams> inDynamicShapeParams = {
    ShapeParams{{{ngraph::Dimension::dynamic(), 100, 4}, {ngraph::Dimension::dynamic(), 5, 100}},
        {{{1, 100, 4}, {1, 5, 100}}, {{2, 100, 4}, {2, 5, 100}}, {{3, 100, 4}, {3, 5, 100}}}, true},
    ShapeParams{{{1, ngraph::Dimension::dynamic(), 4}, {1, 5, ngraph::Dimension::dynamic()}},
        {{{1, 80, 4},  {1, 5, 80}}, {{1, 90, 4}, {1, 5, 90}}, {{1, 100, 4}, {1, 5, 100}}}, false},
    ShapeParams{{{1, 100, 4}, {1, ngraph::Dimension::dynamic(), 100}},
        {{{1, 100, 4}, {1, 5, 100}}, {{1, 100, 4}, {1, 6, 100}}, {{1, 100, 4}, {1, 7, 100}}}, false},
};

const std::vector<int32_t> nmsTopK = {-1, 20};
const std::vector<float> iouThreshold = {0.7f};
const std::vector<float> scoreThreshold = {0.7f};
const std::vector<int32_t> backgroundClass = {-1, 1};
const std::vector<int32_t> keepTopK = {-1, 30};
const std::vector<element::Type> outType = {element::i32, element::i64};

const std::vector<op::v8::MulticlassNms::SortResultType> sortResultType = {
    op::v8::MulticlassNms::SortResultType::SCORE, op::v8::MulticlassNms::SortResultType::CLASSID, op::v8::MulticlassNms::SortResultType::NONE};
const std::vector<bool> sortResDesc = {true, false};
const std::vector<float> nmsEta = {0.6f, 1.0f};
const std::vector<bool> normalized = {true, false};

const auto nmsParamsStatic = ::testing::Combine(
    ::testing::ValuesIn(inStaticShapeParams),
    ::testing::Combine(::testing::Values(ov::element::f32), ::testing::Values(ov::element::i32), ::testing::Values(ov::element::f32)),
    ::testing::ValuesIn(nmsTopK),
    ::testing::Combine(::testing::ValuesIn(iouThreshold), ::testing::ValuesIn(scoreThreshold), ::testing::ValuesIn(nmsEta)),
    ::testing::ValuesIn(backgroundClass),
    ::testing::ValuesIn(keepTopK),
    ::testing::ValuesIn(outType),
    ::testing::ValuesIn(sortResultType),
    ::testing::Combine(::testing::ValuesIn(sortResDesc), ::testing::ValuesIn(normalized)),
    ::testing::Values(CommonTestUtils::DEVICE_CPU));

const auto nmsParamsDynamic = ::testing::Combine(
    ::testing::ValuesIn(inDynamicShapeParams),
    ::testing::Combine(::testing::Values(ov::element::f32), ::testing::Values(ov::element::i32), ::testing::Values(ov::element::f32)),
    ::testing::ValuesIn(nmsTopK),
    ::testing::Combine(::testing::ValuesIn(iouThreshold), ::testing::ValuesIn(scoreThreshold), ::testing::ValuesIn(nmsEta)),
    ::testing::ValuesIn(backgroundClass),
    ::testing::ValuesIn(keepTopK),
    ::testing::ValuesIn(outType),
    ::testing::ValuesIn(sortResultType),
    ::testing::Combine(::testing::ValuesIn(sortResDesc), ::testing::ValuesIn(normalized)),
    ::testing::Values(CommonTestUtils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(smoke_MulticlassNmsLayerTest_static, MulticlassNmsLayerTest, nmsParamsStatic, MulticlassNmsLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MulticlassNmsLayerTest_dynamic, MulticlassNmsLayerTest, nmsParamsDynamic, MulticlassNmsLayerTest::getTestCaseName);