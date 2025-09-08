// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "single_op_tests/roi_align.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::ROIAlignLayerTest;
using ov::test::ROIAlignV9LayerTest;

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

INSTANTIATE_TEST_SUITE_P(smoke_TestsROIAlign_average,
                         ROIAlignLayerTest,
                         ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                                            std::vector<std::vector<ov::Shape>>{{{3, 8, 16, 16}},
                                                                                                {{2, 1, 16, 16}},
                                                                                                {{2, 1, 8, 16}}})),
                                            ::testing::Values(std::vector<size_t>{2, 4}),
                                            ::testing::Values(2),
                                            ::testing::Values(2),
                                            ::testing::ValuesIn(std::vector<float>{1, 0.625}),
                                            ::testing::Values(2),
                                            ::testing::Values("avg"),
                                            ::testing::ValuesIn(netPRCs),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ROIAlignLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TestsROIAlign_max,
                         ROIAlignLayerTest,
                         ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                                            std::vector<std::vector<ov::Shape>>{{{2, 8, 20, 20}},
                                                                                                {{2, 1, 20, 20}},
                                                                                                {{2, 1, 10, 20}}})),
                                            ::testing::Values(std::vector<size_t>{2, 4}),
                                            ::testing::Values(2),
                                            ::testing::Values(2),
                                            ::testing::ValuesIn(std::vector<float>{1, 0.625}),
                                            ::testing::Values(2),
                                            ::testing::Values("max"),
                                            ::testing::ValuesIn(netPRCs),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ROIAlignLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TestsROIAlign_avg_asym,
                         ROIAlignV9LayerTest,
                         ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                                            std::vector<std::vector<ov::Shape>>{{{2, 1, 8, 8}},
                                                                                                {{2, 8, 20, 20}},
                                                                                                {{2, 1, 20, 20}},
                                                                                                {{2, 1, 10, 20}}})),
                                            ::testing::Values(ov::Shape{2, 4}),
                                            ::testing::Values(2),
                                            ::testing::Values(2),
                                            ::testing::ValuesIn(std::vector<float>{1, 0.625}),
                                            ::testing::Values(2),
                                            ::testing::Values("avg"),
                                            ::testing::Values("asymmetric"),
                                            ::testing::ValuesIn(netPRCs),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ROIAlignV9LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TestsROIAlign_avg_hpfn,
                         ROIAlignV9LayerTest,
                         ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                                            std::vector<std::vector<ov::Shape>>{{{2, 1, 8, 8}},
                                                                                                {{2, 8, 20, 20}},
                                                                                                {{2, 1, 20, 20}},
                                                                                                {{2, 1, 10, 20}}})),
                                            ::testing::Values(ov::Shape{2, 4}),
                                            ::testing::Values(2),
                                            ::testing::Values(2),
                                            ::testing::ValuesIn(std::vector<float>{1, 0.625}),
                                            ::testing::Values(2),
                                            ::testing::Values("avg"),
                                            ::testing::Values("half_pixel_for_nn"),
                                            ::testing::ValuesIn(netPRCs),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ROIAlignV9LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TestsROIAlign_max_hp,
                         ROIAlignV9LayerTest,
                         ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                                            std::vector<std::vector<ov::Shape>>{{{2, 1, 8, 8}},
                                                                                                {{2, 8, 20, 20}},
                                                                                                {{2, 1, 20, 20}},
                                                                                                {{2, 1, 10, 20}}})),
                                            ::testing::Values(ov::Shape{2, 4}),
                                            ::testing::Values(2),
                                            ::testing::Values(2),
                                            ::testing::ValuesIn(std::vector<float>{1, 0.625}),
                                            ::testing::Values(2),
                                            ::testing::Values("max"),
                                            ::testing::Values("half_pixel"),
                                            ::testing::ValuesIn(netPRCs),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ROIAlignV9LayerTest::getTestCaseName);
}  // namespace
