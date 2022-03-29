// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/multiclass_nms.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace ov::test::subgraph;
using namespace InferenceEngine;
using namespace ngraph;

const std::vector<std::vector<ov::Shape>> inStaticShapeParams = {
    {{3, 100, 4}, {3,   1, 100}},
    {{1, 10,  4}, {1, 100, 10 }}
};

const std::vector<std::vector<ov::test::InputShape>> inDynamicShapeParams = {
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

const std::vector<int32_t> nmsTopK = {-1, 20};
const std::vector<float> iouThreshold = {0.7f};
const std::vector<float> scoreThreshold = {0.7f};
const std::vector<int32_t> backgroundClass = {-1, 1};
const std::vector<int32_t> keepTopK = {-1, 30};
const std::vector<element::Type> outType = {element::i32, element::i64};

const std::vector<ov::op::util::MulticlassNmsBase::SortResultType> sortResultType = {
    ov::op::util::MulticlassNmsBase::SortResultType::SCORE, ov::op::util::MulticlassNmsBase::SortResultType::CLASSID, ov::op::util::MulticlassNmsBase::SortResultType::NONE};
const std::vector<bool> sortResDesc = {true, false};
const std::vector<float> nmsEta = {0.6f, 1.0f};
const std::vector<bool> normalized = {true, false};

const auto nmsParamsStatic = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inStaticShapeParams)),
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