// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/roi_align.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::ROIAlignLayerTest;

namespace {

const std::vector<ov::element::Type> model_types = {
    ov::element::f16,
    ov::element::f32
};

const auto ROIAlignCases_average = ::testing::Combine(
        ::testing::ValuesIn(
    ov::test::static_shapes_to_test_representation(
        std::vector<std::vector<ov::Shape>>{
            {{ 3, 8, 16, 16 }},
            {{ 2, 1, 16, 16 }},
            {{ 2, 1, 8, 16 }}})),
        ::testing::Values(ov::Shape{ 2, 4 }),
        ::testing::Values(2),
        ::testing::Values(2),
        ::testing::ValuesIn(std::vector<float> { 1, 0.625 }),
        ::testing::Values(2),
        ::testing::Values("avg"),
        ::testing::ValuesIn(model_types),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsROIAlign_average, ROIAlignLayerTest, ROIAlignCases_average, ROIAlignLayerTest::getTestCaseName);

const auto ROIAlignCases_max = ::testing::Combine(
        ::testing::ValuesIn(
        ov::test::static_shapes_to_test_representation(
            std::vector<std::vector<ov::Shape>>{
            {{ 2, 8, 20, 20 }},
            {{ 2, 1, 20, 20 }},
            {{ 2, 1, 10, 20 }}})),
        ::testing::Values(ov::Shape{ 2, 4 }),
        ::testing::Values(2),
        ::testing::Values(2),
        ::testing::ValuesIn(std::vector<float> { 1, 0.625 }),
        ::testing::Values(2),
        ::testing::Values("max"),
        ::testing::ValuesIn(model_types),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsROIAlign_max, ROIAlignLayerTest, ROIAlignCases_max, ROIAlignLayerTest::getTestCaseName);

}  // namespace
