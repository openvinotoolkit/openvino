// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <tuple>

#include "single_layer_tests/matrix_nms.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine;
using namespace ngraph;
const std::vector<InputShapeParams> inStaticShapeParams = {
        // dynamic shape, {{batch, box, 4}, {batch, class, box}}
        {{}, {{{3, 100, 4}, {3,   5, 100}}}},
        {{}, {{{1, 10,  4}, {1, 100, 10 }}}},
        {{}, {{{2, 50,  4}, {2,  50, 50 }}}},
};

const std::vector<InputShapeParams> inDynamicShapeParams = {
        {{{ngraph::Dimension::dynamic(), 100, 4}, {ngraph::Dimension::dynamic(), 5, 100}},
            {{{1, 100, 4}, {1, 5, 100}}, {{2, 100, 4}, {2, 5, 100}}, {{3, 100, 4}, {3, 5, 100}}}},
        {{{1, ngraph::Dimension::dynamic(), 4}, {1, 5, ngraph::Dimension::dynamic()}},
            {{{1, 80, 4},  {1, 5, 80}}, {{1, 90, 4}, {1, 5, 90}}, {{1, 100, 4}, {1, 5, 100}}}},
        {{{1, 100, 4}, {1, ngraph::Dimension::dynamic(), 100}},
            {{{1, 100, 4}, {1, 5, 100}}, {{1, 100, 4}, {1, 6, 100}}, {{1, 100, 4}, {1, 7, 100}}}},
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
const std::vector<int> backgroudClass = {-1, 0};
const std::vector<bool> normalized = {true, false};
const std::vector<op::v8::MatrixNms::DecayFunction> decayFunction = {op::v8::MatrixNms::DecayFunction::GAUSSIAN,
                                                op::v8::MatrixNms::DecayFunction::LINEAR};

const auto nmsParamsStatic = ::testing::Combine(::testing::ValuesIn(inStaticShapeParams),
                                                ::testing::Combine(::testing::Values(Precision::FP32),
                                                                    ::testing::Values(Precision::I32),
                                                                    ::testing::Values(Precision::FP32)),
                                                ::testing::ValuesIn(sortResultType),
                                                ::testing::ValuesIn(outType),
                                                ::testing::ValuesIn(topKParams),
                                                ::testing::ValuesIn(thresholdParams),
                                                ::testing::ValuesIn(backgroudClass),
                                                ::testing::ValuesIn(normalized),
                                                ::testing::ValuesIn(decayFunction),
                                                ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto nmsParamsDynamic = ::testing::Combine(::testing::ValuesIn(inDynamicShapeParams),
                                                 ::testing::Combine(::testing::Values(Precision::FP32),
                                                                    ::testing::Values(Precision::I32),
                                                                    ::testing::Values(Precision::FP32)),
                                                 ::testing::ValuesIn(sortResultType),
                                                 ::testing::ValuesIn(outType),
                                                 ::testing::ValuesIn(topKParams),
                                                 ::testing::ValuesIn(thresholdParams),
                                                 ::testing::ValuesIn(backgroudClass),
                                                 ::testing::ValuesIn(normalized),
                                                 ::testing::ValuesIn(decayFunction),
                                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_MatrixNmsLayerTest_static, MatrixNmsLayerTest, nmsParamsStatic, MatrixNmsLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MatrixNmsLayerTest_dynamic, MatrixNmsLayerTest, nmsParamsDynamic, MatrixNmsLayerTest::getTestCaseName);
