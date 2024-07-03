// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/non_max_suppression.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::Nms9LayerTest;

const std::vector<ov::test::InputShapeParams> inShapeParams = {
    {2, 50, 50},
    {9, 10, 10}  // GPU implementation is tested on this shape only
};

const std::vector<int32_t> maxOutBoxPerClass = {5, 20};
const std::vector<float> threshold = {0.3f, 0.7f};
const std::vector<float> sigmaThreshold = {0.0f, 0.5f};
const std::vector<ov::op::v5::NonMaxSuppression::BoxEncodingType> encodType = {ov::op::v5::NonMaxSuppression::BoxEncodingType::CENTER,
                                                                               ov::op::v5::NonMaxSuppression::BoxEncodingType::CORNER};
const std::vector<bool> sortResDesc = {true, false};
const std::vector<ov::element::Type> outType = {ov::element::i32};//, ov::element::i64};

const std::vector<ov::element::Type> input_types = {ov::element::f32, ov::element::f16};

INSTANTIATE_TEST_SUITE_P(smoke_Nms9LayerTest,
                         Nms9LayerTest,
                         ::testing::Combine(::testing::ValuesIn(inShapeParams),
                                            ::testing::Combine(::testing::ValuesIn(input_types),
                                                               ::testing::Values(ov::element::i32),
                                                               ::testing::Values(ov::element::f32)),
                                            ::testing::ValuesIn(maxOutBoxPerClass),
                                            ::testing::ValuesIn(threshold),
                                            ::testing::ValuesIn(threshold),
                                            ::testing::ValuesIn(sigmaThreshold),
                                            ::testing::ValuesIn(encodType),
                                            ::testing::ValuesIn(sortResDesc),
                                            ::testing::ValuesIn(outType),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         Nms9LayerTest::getTestCaseName);
} // namespace
