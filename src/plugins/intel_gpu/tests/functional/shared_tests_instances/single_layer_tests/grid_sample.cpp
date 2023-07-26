// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/grid_sample.hpp"

#include <vector>

using namespace LayerTestsDefinitions;

using GridSampleOp = ov::op::v9::GridSample;

namespace {

const std::vector<std::vector<size_t>> data_shapes = {
    {5, 2, 3, 5},  // Odd
    {5, 3, 4, 6},  // Even
};

const std::vector<std::vector<size_t>> grid_shapes = {
    {5, 7, 3, 2},  // Odd
    {5, 2, 8, 2},  // Even
};

const std::vector<bool> align_corners = {true, false};

const std::vector<GridSampleOp::InterpolationMode> modes = {
    GridSampleOp::InterpolationMode::BILINEAR,
    GridSampleOp::InterpolationMode::BICUBIC,
    GridSampleOp::InterpolationMode::NEAREST,
};

const std::vector<GridSampleOp::PaddingMode> padding_modes = {
    GridSampleOp::PaddingMode::ZEROS,
    GridSampleOp::PaddingMode::BORDER,
    GridSampleOp::PaddingMode::REFLECTION,
};

const std::vector<InferenceEngine::Precision> data_precisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

const std::vector<InferenceEngine::Precision> grid_precisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
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
