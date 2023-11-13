// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/multiclass_nms.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace ov::test::subgraph;
using namespace InferenceEngine;
using namespace ngraph;

/* input format #1 with 2 inputs: bboxes N, M, 4, scores N, C, M */
const std::vector<std::vector<ov::Shape>> shapes2Inputs = {
    {{3, 100, 4}, {3,   1, 100}},
    {{1, 10,  4}, {1, 100, 10 }}
};

/* input format #2 with 3 inputs: bboxes C, M, 4, scores C, M, roisnum N */
const std::vector<std::vector<ov::Shape>> shapes3Inputs = {
    {{1, 10, 4}, {1, 10}, {1}},
    {{1, 10, 4}, {1, 10}, {10}},
    {{2, 100, 4}, {2, 100}, {1}},
    {{2, 100, 4}, {2, 100}, {10}}
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

const std::vector<bool> outStaticShape = {true};   // only be true as gpu plugin not support nms with internal dynamic yet.

const auto params_v9_2Inputs = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes2Inputs)),
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
    ::testing::Values(ov::test::utils::DEVICE_GPU));


INSTANTIATE_TEST_SUITE_P(smoke_MulticlassNmsLayerTest_v9_2inputs,
                         MulticlassNmsLayerTest,
                         params_v9_2Inputs,
                         MulticlassNmsLayerTest::getTestCaseName);

const auto params_v9_3Inputs = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes3Inputs)),
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
    ::testing::Values(ov::test::utils::DEVICE_GPU));

INSTANTIATE_TEST_SUITE_P(smoke_MulticlassNmsLayerTest_v9_3inputs,
                         MulticlassNmsLayerTest,
                         params_v9_3Inputs,
                         MulticlassNmsLayerTest::getTestCaseName);

const auto params_v8 = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes2Inputs)),
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
    ::testing::Values(ov::test::utils::DEVICE_GPU));


INSTANTIATE_TEST_SUITE_P(smoke_MulticlassNmsLayerTest_v8,
                         MulticlassNmsLayerTest8,
                         params_v8,
                         MulticlassNmsLayerTest8::getTestCaseName);
