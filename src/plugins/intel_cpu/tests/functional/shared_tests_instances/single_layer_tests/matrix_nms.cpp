// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <tuple>

#include "single_op_tests/matrix_nms.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::MatrixNmsLayerTest;

const std::vector<std::vector<ov::Shape>> inStaticShapeParams = {
    {{3, 100, 4}, {3,   1, 100}},
    {{1, 10,  4}, {1, 100, 10 }}
};

const std::vector<std::vector<ov::test::InputShape>> inDynamicShapeParams = {
    // num_batches, num_boxes, 4
    {{{ov::Dimension::dynamic(), ov::Dimension::dynamic(), 4},
        {{1, 10, 4}, {2, 100, 4}}},
    // num_batches, num_classes, num_boxes
     {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()},
        {{1, 3, 10}, {2, 5, 100}}}},
    {{{ov::Dimension(1, 10), ov::Dimension(1, 100), 4},
        {{1, 10, 4}, {2, 100, 4}}},
    {{{ov::Dimension(1, 10), ov::Dimension(1, 100), ov::Dimension(1, 100)}},
        {{1, 3, 10}, {2, 5, 100}}}}
};

const std::vector<ov::op::v8::MatrixNms::SortResultType> sortResultType = {ov::op::v8::MatrixNms::SortResultType::CLASSID,
                                                                       ov::op::v8::MatrixNms::SortResultType::SCORE,
                                                                       ov::op::v8::MatrixNms::SortResultType::NONE};
const std::vector<ov::element::Type> outType = {ov::element::i32, ov::element::i64};
const std::vector<ov::test::TopKParams> topKParams = {
    {-1, 5},
    {100, -1}
};
const std::vector<ov::test::ThresholdParams> thresholdParams = {
    {0.0f, 2.0f, 0.0f},
    {0.1f, 1.5f, 0.2f}
};
const std::vector<int> nmsTopK = {-1, 100};
const std::vector<int> keepTopK = {-1, 5};
const std::vector<int> backgroudClass = {-1, 1};
const std::vector<bool> normalized = {true, false};
const std::vector<ov::op::v8::MatrixNms::DecayFunction> decayFunction = {ov::op::v8::MatrixNms::DecayFunction::GAUSSIAN,
                                                ov::op::v8::MatrixNms::DecayFunction::LINEAR};

const auto nmsParamsStatic = ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inStaticShapeParams)),
                                                ::testing::Values(ov::element::f32),
                                                ::testing::ValuesIn(sortResultType),
                                                ::testing::ValuesIn(outType),
                                                ::testing::ValuesIn(topKParams),
                                                ::testing::ValuesIn(thresholdParams),
                                                ::testing::ValuesIn(backgroudClass),
                                                ::testing::ValuesIn(normalized),
                                                ::testing::ValuesIn(decayFunction),
                                                ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto nmsParamsDynamic = ::testing::Combine(::testing::ValuesIn(inDynamicShapeParams),
                                                 ::testing::Values(ov::element::f32),
                                                 ::testing::ValuesIn(sortResultType),
                                                 ::testing::ValuesIn(outType),
                                                 ::testing::ValuesIn(topKParams),
                                                 ::testing::ValuesIn(thresholdParams),
                                                 ::testing::ValuesIn(backgroudClass),
                                                 ::testing::ValuesIn(normalized),
                                                 ::testing::ValuesIn(decayFunction),
                                                 ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_MatrixNmsLayerTest_static, MatrixNmsLayerTest, nmsParamsStatic, MatrixNmsLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MatrixNmsLayerTest_dynamic, MatrixNmsLayerTest, nmsParamsDynamic, MatrixNmsLayerTest::getTestCaseName);
} // namespace
