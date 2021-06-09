// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <tuple>

#include "single_layer_tests/multiclass_nms.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine;
using namespace ngraph;

const std::vector<InputShapeParams> inShapeParams = {
    InputShapeParams{3, 100, 5},
    InputShapeParams{1, 10, 50},
    InputShapeParams{2, 50, 50}
};

const std::vector<op::v8::MulticlassNms::SortResultType> sortResultType = {op::v8::MulticlassNms::SortResultType::CLASSID,
                                                                       op::v8::MulticlassNms::SortResultType::SCORE,
                                                                       op::v8::MulticlassNms::SortResultType::NONE};
const std::vector<element::Type> outType = {element::i32, element::i64};
const std::vector<int> nmsTopK = {10, 100};
const std::vector<int> keepTopK = {10, 5};
const std::vector<float> iouThr = {0.1f, 0.6f};
const std::vector<float> scoreThr = {0.3f, 0.7f};
const std::vector<int> backgroudClass = {-1, 0};
const std::vector<float> eta = {0.0f, 0.5f};

const auto nmsParams = ::testing::Combine(::testing::ValuesIn(inShapeParams),
                                          ::testing::Combine(::testing::Values(Precision::FP32),
                                                             ::testing::Values(Precision::I32),
                                                             ::testing::Values(Precision::FP32)),
                                          ::testing::ValuesIn(sortResultType),
                                          ::testing::ValuesIn(outType),
                                          ::testing::ValuesIn(iouThr),
                                          ::testing::ValuesIn(nmsTopK),
                                          ::testing::ValuesIn(keepTopK),
                                          ::testing::ValuesIn(backgroudClass),
                                          ::testing::ValuesIn(eta),
                                          ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(smoke_MulticlassNmsLayerTest, MulticlassNmsLayerTest, nmsParams, MulticlassNmsLayerTest::getTestCaseName);
