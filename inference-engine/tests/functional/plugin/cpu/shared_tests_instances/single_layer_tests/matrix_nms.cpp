// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <tuple>

#include "single_layer_tests/matrix_nms.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine;
using namespace ngraph;

const std::vector<InputShapeParams> inShapeParams = {
    InputShapeParams{3, 100, 5},
    InputShapeParams{1, 10, 50},
    InputShapeParams{2, 50, 50}
};

const std::vector<op::v8::MatrixNms::SortResultType> sortResultType = {op::v8::MatrixNms::SortResultType::CLASSID,
                                                                       op::v8::MatrixNms::SortResultType::SCORE,
                                                                       op::v8::MatrixNms::SortResultType::NONE};
const std::vector<element::Type> outType = {element::i32, element::i64};
const std::vector<TopKParams> topKParams = {
    {-1, 5},
    {100, -1}
};
const std::vector<ThresholdParams> thresholdParams = {
    {0.0f, 2.0f, 0.0f},
    {0.1f, 1.5f, 0.2f}
};
const std::vector<int> nmsTopK = {-1, 100};
const std::vector<int> keepTopK = {-1, 5};
const std::vector<int> backgroudClass = {-1, 0};
const std::vector<bool> normalized = {true, false};
const std::vector<op::v8::MatrixNms::DecayFunction> decayFunction = {op::v8::MatrixNms::DecayFunction::GAUSSIAN,
                                                op::v8::MatrixNms::DecayFunction::LINEAR};

const auto nmsParams = ::testing::Combine(::testing::ValuesIn(inShapeParams),
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

INSTANTIATE_TEST_CASE_P(smoke_MatrixNmsLayerTest, MatrixNmsLayerTest, nmsParams, MatrixNmsLayerTest::getTestCaseName);
