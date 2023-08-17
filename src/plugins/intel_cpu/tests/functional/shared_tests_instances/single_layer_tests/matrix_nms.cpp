// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <tuple>

#include "single_layer_tests/matrix_nms.hpp"
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

const std::vector<op::v8::MatrixNms::SortResultType> sortResultType = {op::v8::MatrixNms::SortResultType::CLASSID,
                                                                       op::v8::MatrixNms::SortResultType::SCORE,
                                                                       op::v8::MatrixNms::SortResultType::NONE};
const std::vector<element::Type> outType = {element::i32, element::i64};
const std::vector<TopKParams> topKParams = {
    TopKParams{-1, 5},
    TopKParams{100, -1}
};
const std::vector<ThresholdParams> thresholdParams = {
    ThresholdParams{0.0f, 2.0f, 0.0f},
    ThresholdParams{0.1f, 1.5f, 0.2f}
};
const std::vector<int> nmsTopK = {-1, 100};
const std::vector<int> keepTopK = {-1, 5};
const std::vector<int> backgroudClass = {-1, 1};
const std::vector<bool> normalized = {true, false};
const std::vector<op::v8::MatrixNms::DecayFunction> decayFunction = {op::v8::MatrixNms::DecayFunction::GAUSSIAN,
                                                op::v8::MatrixNms::DecayFunction::LINEAR};

const std::vector<bool> outStaticShape = {false};   // always be false for cpu plugin with ov2.0.

const auto nmsParamsStatic = ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inStaticShapeParams)),
                                                ::testing::Combine(::testing::Values(ov::element::f32),
                                                                   ::testing::Values(ov::element::i32),
                                                                   ::testing::Values(ov::element::f32)),
                                                ::testing::ValuesIn(sortResultType),
                                                ::testing::ValuesIn(outType),
                                                ::testing::ValuesIn(topKParams),
                                                ::testing::ValuesIn(thresholdParams),
                                                ::testing::ValuesIn(backgroudClass),
                                                ::testing::ValuesIn(normalized),
                                                ::testing::ValuesIn(decayFunction),
                                                ::testing::ValuesIn(outStaticShape),
                                                ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto nmsParamsDynamic = ::testing::Combine(::testing::ValuesIn(inDynamicShapeParams),
                                                 ::testing::Combine(::testing::Values(ov::element::f32),
                                                                    ::testing::Values(ov::element::i32),
                                                                    ::testing::Values(ov::element::f32)),
                                                 ::testing::ValuesIn(sortResultType),
                                                 ::testing::ValuesIn(outType),
                                                 ::testing::ValuesIn(topKParams),
                                                 ::testing::ValuesIn(thresholdParams),
                                                 ::testing::ValuesIn(backgroudClass),
                                                 ::testing::ValuesIn(normalized),
                                                 ::testing::ValuesIn(decayFunction),
                                                ::testing::ValuesIn(outStaticShape),
                                                 ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_MatrixNmsLayerTest_static, MatrixNmsLayerTest, nmsParamsStatic, MatrixNmsLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MatrixNmsLayerTest_dynamic, MatrixNmsLayerTest, nmsParamsDynamic, MatrixNmsLayerTest::getTestCaseName);
