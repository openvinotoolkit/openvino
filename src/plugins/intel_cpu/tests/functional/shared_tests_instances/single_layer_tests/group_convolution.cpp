// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_op_tests/group_convolution.hpp"

namespace {
using ov::test::GroupConvolutionLayerTest;

const std::vector<ov::element::Type> model_types = {
    ov::element::f32,
    ov::element::f16,
    ov::element::i32};

/* ============= _1d GroupConvolution ============= */
const std::vector<ov::Shape> input_shapes_1d = {{1, 16, 30}};

const std::vector<std::vector<size_t>> kernels_1d = {{3}};
const std::vector<std::vector<size_t>> strides_1d = {{1}};
const std::vector<std::vector<ptrdiff_t>> pad_begins_1d = {{0}};
const std::vector<std::vector<ptrdiff_t>> pad_ends_1d = {{0}};
const std::vector<std::vector<size_t>> dilations_1d = {{1}};
const std::vector<size_t> num_out_channels_1d = {8, 16};
const std::vector<size_t> num_groups_1d = {2, 8};

const auto group_conv_1d_params_explicit_padding = ::testing::Combine(
    ::testing::ValuesIn(kernels_1d),
    ::testing::ValuesIn(strides_1d),
    ::testing::ValuesIn(pad_begins_1d),
    ::testing::ValuesIn(pad_ends_1d),
    ::testing::ValuesIn(dilations_1d),
    ::testing::ValuesIn(num_out_channels_1d),
    ::testing::ValuesIn(num_groups_1d),
    ::testing::Values(ov::op::PadType::EXPLICIT));

const auto group_conv_1d_params_auto_pad_valid = ::testing::Combine(
    ::testing::ValuesIn(kernels_1d), ::testing::ValuesIn(strides_1d),
    ::testing::Values(std::vector<ptrdiff_t>({0})),
    ::testing::Values(std::vector<ptrdiff_t>({0})),
    ::testing::ValuesIn(dilations_1d), ::testing::ValuesIn(num_out_channels_1d),
    ::testing::ValuesIn(num_groups_1d),
    ::testing::Values(ov::op::PadType::VALID));

INSTANTIATE_TEST_SUITE_P(
    smoke_GroupConvolution1D_ExplicitPadding, GroupConvolutionLayerTest,
    ::testing::Combine(
        group_conv_1d_params_explicit_padding,
        ::testing::ValuesIn(model_types),
        ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_1d)),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
    GroupConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_GroupConvolution1D_AutoPadValid, GroupConvolutionLayerTest,
    ::testing::Combine(
        group_conv_1d_params_auto_pad_valid,
        ::testing::ValuesIn(model_types),
        ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_1d)),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
    GroupConvolutionLayerTest::getTestCaseName);

/* ============= 2D GroupConvolution ============= */
const std::vector<ov::Shape> input_shapes_2d = {{1, 16, 30, 30}};

const std::vector<std::vector<size_t>> kernels_2d = {{3, 3}};
const std::vector<std::vector<size_t>> strides_2d = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> pad_begins_2d = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> pad_ends_2d = {{0, 0}};
const std::vector<std::vector<size_t>> dilations_2d = {{1, 1}};
const std::vector<size_t> num_out_channels_2d = {8, 16};
const std::vector<size_t> num_groups_2d = {2, 8};

const auto group_conv2DParams_explicit_padding = ::testing::Combine(
    ::testing::ValuesIn(kernels_2d),
    ::testing::ValuesIn(strides_2d),
    ::testing::ValuesIn(pad_begins_2d),
    ::testing::ValuesIn(pad_ends_2d),
    ::testing::ValuesIn(dilations_2d),
    ::testing::ValuesIn(num_out_channels_2d),
    ::testing::ValuesIn(num_groups_2d),
    ::testing::Values(ov::op::PadType::EXPLICIT));

const auto group_conv2DParams_auto_pad_valid = ::testing::Combine(
    ::testing::ValuesIn(kernels_2d),
    ::testing::ValuesIn(strides_2d),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::ValuesIn(dilations_2d),
    ::testing::ValuesIn(num_out_channels_2d),
    ::testing::ValuesIn(num_groups_2d),
    ::testing::Values(ov::op::PadType::VALID));

INSTANTIATE_TEST_SUITE_P(
    smoke_GroupConvolution2D_ExplicitPadding, GroupConvolutionLayerTest,
    ::testing::Combine(
        group_conv2DParams_explicit_padding,
        ::testing::ValuesIn(model_types),
        ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_2d)),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
    GroupConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_GroupConvolution2D_AutoPadValid, GroupConvolutionLayerTest,
    ::testing::Combine(
        group_conv2DParams_auto_pad_valid,
        ::testing::ValuesIn(model_types),
        ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_2d)),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
    GroupConvolutionLayerTest::getTestCaseName);

/* ============= 3D GroupConvolution ============= */
const std::vector<ov::Shape> input_shapes_3d = {{1, 4, 10, 10, 10}};

const std::vector<std::vector<size_t>> kernels_3d = {{3, 3, 3}};
const std::vector<std::vector<ptrdiff_t>> paddings_3d = {{0, 0, 0}};
const std::vector<std::vector<size_t>> strides_3d = {{1, 1, 1}};
const std::vector<std::vector<size_t>> dilations_3d = {{1, 1, 1}};

const auto group_conv3DParams_explicit_padding = ::testing::Combine(
    ::testing::ValuesIn(kernels_3d),
    ::testing::ValuesIn(strides_3d),
    ::testing::ValuesIn(paddings_3d),
    ::testing::ValuesIn(paddings_3d),
    ::testing::ValuesIn(dilations_3d),
    ::testing::Values(4),
    ::testing::Values(2),
    ::testing::Values(ov::op::PadType::EXPLICIT));

const auto group_conv3DParams_auto_pad_valid = ::testing::Combine(
    ::testing::ValuesIn(kernels_3d),
    ::testing::ValuesIn(strides_3d),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
    ::testing::ValuesIn(dilations_3d),
    ::testing::Values(4),
    ::testing::Values(2),
    ::testing::Values(ov::op::PadType::VALID));

INSTANTIATE_TEST_SUITE_P(
    smoke_GroupConvolution3D_ExplicitPadding, GroupConvolutionLayerTest,
    ::testing::Combine(
        group_conv3DParams_explicit_padding,
        ::testing::ValuesIn(model_types),
        ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_3d)),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
    GroupConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_GroupConvolution3D_AutoPadValid, GroupConvolutionLayerTest,
    ::testing::Combine(
        group_conv3DParams_auto_pad_valid,
        ::testing::ValuesIn(model_types),
        ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_3d)),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
    GroupConvolutionLayerTest::getTestCaseName);

}  // namespace
