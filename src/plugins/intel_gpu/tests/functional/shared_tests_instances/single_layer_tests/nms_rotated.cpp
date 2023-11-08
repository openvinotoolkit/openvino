// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/nms_rotated.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine;
using namespace ngraph;

const std::vector<InputShapeParams> inShapeParams = {
    InputShapeParams{2, 50, 50},
    InputShapeParams {9, 10, 10}
};

const std::vector<int32_t> maxOutBoxPerClass = {5, 20};
const std::vector<float> threshold = {0.3f, 0.7f};
const std::vector<bool> sortResDesc = {true, false};
const std::vector<element::Type> outType = {element::i32, element::i64};
const std::vector<bool> clockwise = {true, false};

const std::vector<Precision> inputPrecisions = {Precision::FP32, Precision::FP16};

INSTANTIATE_TEST_SUITE_P(smoke_NmsRotatedLayerTest,
                         NmsRotatedLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inShapeParams),
                                            ::testing::Combine(::testing::ValuesIn(inputPrecisions),
                                                               ::testing::Values(Precision::I32),
                                                               ::testing::Values(Precision::FP32)),
                                            ::testing::ValuesIn(maxOutBoxPerClass),
                                            ::testing::ValuesIn(threshold),
                                            ::testing::ValuesIn(threshold),
                                            ::testing::ValuesIn(sortResDesc),
                                            ::testing::ValuesIn(outType),
                                            ::testing::ValuesIn(clockwise),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         NmsRotatedLayerTest::getTestCaseName);
