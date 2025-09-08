// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/grid_sample.hpp"

namespace {
using ov::op::v9::GridSample;
using ov::test::GridSampleLayerTest;

const std::vector<ov::Shape> data_shapes = {
    {5, 2, 3, 5},  // Odd
    {5, 3, 4, 6},  // Even
};

const std::vector<ov::Shape> grid_shapes = {
    {5, 7, 3, 2},  // Odd
    {5, 2, 8, 2},  // Even
};

const std::vector<bool> align_corners = {true, false};

const std::vector<GridSample::InterpolationMode> modes = {
    GridSample::InterpolationMode::BILINEAR,
    GridSample::InterpolationMode::BICUBIC,
    GridSample::InterpolationMode::NEAREST,
};

const std::vector<GridSample::PaddingMode> padding_modes = {
    GridSample::PaddingMode::ZEROS,
    GridSample::PaddingMode::BORDER,
    GridSample::PaddingMode::REFLECTION,
};

const std::vector<ov::element::Type> data_precisions = {
    ov::element::f32,
    ov::element::f16,
};

const std::vector<ov::element::Type> grid_precisions = {
    ov::element::f32,
};

INSTANTIATE_TEST_SUITE_P(smoke_GridSample,
                         GridSampleLayerTest,
                         testing::Combine(testing::ValuesIn(data_shapes),
                                          testing::ValuesIn(grid_shapes),
                                          testing::ValuesIn(align_corners),
                                          testing::ValuesIn(modes),
                                          testing::ValuesIn(padding_modes),
                                          testing::ValuesIn(data_precisions),
                                          testing::ValuesIn(grid_precisions),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         GridSampleLayerTest::getTestCaseName);
}  // namespace
