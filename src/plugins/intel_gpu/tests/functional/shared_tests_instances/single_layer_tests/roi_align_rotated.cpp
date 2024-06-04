// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "single_op_tests/roi_align_rotated.hpp"

#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::ROIAlignRotatedLayerTest;

const std::vector<ov::element::Type> netPRCs = {
    ov::element::f32
    // There is no possibility to test ROIAlign in fp16 precision,
    // because on edge cases where in fp32 version ROI value is
    // a little bit smaller than the nearest integer value,
    // it would be bigger than the nearest integer in fp16 precision.
    // Such behavior leads to completely different results of ROIAlign
    // in fp32 and fp16 precisions.
    // In real AI applications this problem is solved by precision-aware training.

    // ov::element::f16
};

INSTANTIATE_TEST_SUITE_P(gtest_smoke_TestsROIAlignRotatedROIAlignLayerTest_EvalGenerateName_,
                         ROIAlignRotatedLayerTest,
                         ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                                std::vector<std::vector<ov::Shape>>{{{3, 8, 16, 16}},
                                                                                    {{2, 1, 16, 10}},
                                                                                    {{4, 3, 5, 12}}})),
                                            ::testing::ValuesIn(std::vector<int>{2, 4}),
                                            ::testing::Values(2),
                                            ::testing::Values(2),
                                            ::testing::Values(2),
                                            ::testing::ValuesIn(std::vector<float>{1, 0.625}),
                                            ::testing::ValuesIn(std::vector<bool>{true, false}),
                                            ::testing::ValuesIn(netPRCs),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ROIAlignRotatedLayerTest::getTestCaseName);

}  // namespace
