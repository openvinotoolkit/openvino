// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/matrix_nms.hpp"

#include <tuple>
#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace ov::test::subgraph;
using namespace InferenceEngine;
using namespace ngraph;

const std::vector<std::vector<ov::Shape>> inStaticShapeParams = {{{3, 100, 4}, {3, 1, 100}},
                                                                 {{1, 10, 4}, {1, 100, 10}}};

const auto inputPrecisions = InputPrecisions{ov::element::f32, ov::element::i32, ov::element::f32};

const std::vector<op::v8::MatrixNms::SortResultType> sortResultType = {op::v8::MatrixNms::SortResultType::CLASSID,
                                                                       op::v8::MatrixNms::SortResultType::SCORE,
                                                                       op::v8::MatrixNms::SortResultType::NONE};
const std::vector<element::Type> outType = {element::i32, element::i64};
const std::vector<TopKParams> topKParams = {TopKParams{-1, 5}, TopKParams{100, -1}};
const std::vector<ThresholdParams> thresholdParams = {ThresholdParams{0.0f, 2.0f, 0.0f},
                                                      ThresholdParams{0.1f, 1.5f, 0.2f}};
const std::vector<int> backgroudClass = {-1, 1};
const std::vector<bool> normalized = {true, false};
const std::vector<op::v8::MatrixNms::DecayFunction> decayFunction = {op::v8::MatrixNms::DecayFunction::GAUSSIAN,
                                                                     op::v8::MatrixNms::DecayFunction::LINEAR};

const std::vector<bool> outStaticShape = {true};   // only be true as gpu plugin not support nms with internal dynamic yet.

const auto nmsParamsStatic =
    ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inStaticShapeParams)),
                       ::testing::Values(inputPrecisions),
                       ::testing::ValuesIn(sortResultType),
                       ::testing::ValuesIn(outType),
                       ::testing::ValuesIn(topKParams),
                       ::testing::ValuesIn(thresholdParams),
                       ::testing::ValuesIn(backgroudClass),
                       ::testing::ValuesIn(normalized),
                       ::testing::ValuesIn(decayFunction),
                       ::testing::ValuesIn(outStaticShape),
                       ::testing::Values(ov::test::utils::DEVICE_GPU));

INSTANTIATE_TEST_SUITE_P(smoke_MatrixNmsLayerTest_static,
                         MatrixNmsLayerTest,
                         nmsParamsStatic,
                         MatrixNmsLayerTest::getTestCaseName);
