// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "single_op_tests/deformable_convolution.hpp"

namespace {
using ov::test::DeformableConvolutionLayerTest;
const std::vector<ov::element::Type> netPrecisions = {
    ov::element::f32,
    ov::element::f16
};

/* ============= 2D DeformableConvolution ============= */
const std::vector<std::vector<size_t>> deformable_values_0 = {{1, 16, 2, 2}};
const std::vector<std::vector<size_t>> kernel_shapes_0 = {{2, 2, 2, 2}};
const std::vector<std::vector<size_t>> stride_values = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> pad_begin_values = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> pad_end_values ={{0, 0}};
const std::vector<std::vector<size_t>> dilation_values = {{1, 1}};
const std::vector<size_t> num_groups = {1};
const std::vector<size_t> num_deformable_groups_0 = {2};
const std::vector<size_t> num_out_channels = {1, 5};
const std::vector<size_t> multiple_defor_groups = {4};
const std::vector<std::vector<size_t>> deformable_values_1 = {{1, 72, 64, 64}};
const std::vector<std::vector<size_t>> kernel_shapes_1 = {{16, 16, 3, 3}};

const std::vector<bool> with_bilinear_interpolation_pad = { false, true };
const std::vector<bool> with_modulated_scalar = { false, true };

const auto deformableConv2DParams_ExplicitPadding = ::testing::Combine(
    ::testing::ValuesIn(stride_values),
    ::testing::ValuesIn(pad_begin_values),
    ::testing::ValuesIn(pad_end_values),
    ::testing::ValuesIn(dilation_values),
    ::testing::ValuesIn(num_groups),
    ::testing::ValuesIn(num_deformable_groups_0),
    ::testing::ValuesIn(num_out_channels),
    ::testing::Values(ov::op::PadType::EXPLICIT),
    ::testing::ValuesIn(with_bilinear_interpolation_pad));

const auto deformableConv2DParams_AutoPadValid = ::testing::Combine(
    ::testing::ValuesIn(stride_values),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::ValuesIn(dilation_values),
    ::testing::ValuesIn(num_groups),
    ::testing::ValuesIn(num_deformable_groups_0),
    ::testing::ValuesIn(num_out_channels),
    ::testing::Values(ov::op::PadType::VALID),
    ::testing::ValuesIn(with_bilinear_interpolation_pad));

const auto deformableConv2DParams_DeformableGroups_AutoPadExplicit = ::testing::Combine(
    ::testing::ValuesIn(stride_values),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::ValuesIn(dilation_values),
    ::testing::ValuesIn(num_groups),
    ::testing::ValuesIn(multiple_defor_groups),
    ::testing::ValuesIn(num_out_channels),
    ::testing::Values(ov::op::PadType::EXPLICIT),
    ::testing::ValuesIn(with_bilinear_interpolation_pad));

INSTANTIATE_TEST_SUITE_P(
    smoke_DeformableConvolution2D_ExplicitPadding, DeformableConvolutionLayerTest,
    ::testing::Combine(
        deformableConv2DParams_ExplicitPadding,
        ::testing::ValuesIn(with_modulated_scalar),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{1, 2, 3, 3}, {1, 16, 2, 2}, {2, 2, 2, 2}, {1, 8, 2, 2}}))),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    DeformableConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_DeformableConvolution2D_AutoPadValid, DeformableConvolutionLayerTest,
    ::testing::Combine(
        deformableConv2DParams_AutoPadValid,
        ::testing::ValuesIn(with_modulated_scalar),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{1, 2, 3, 3}, {1, 16, 2, 2}, {2, 2, 2, 2}, {1, 8, 2, 2}}))),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    DeformableConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_DeformableConvolution2D_DeformableGroups_ExplicitPadding, DeformableConvolutionLayerTest,
    ::testing::Combine(
        deformableConv2DParams_DeformableGroups_AutoPadExplicit,
        ::testing::ValuesIn(with_modulated_scalar),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{1, 16, 66, 66},
                                                                                                 {1, 72, 64, 64},
                                                                                                 {16, 16, 3, 3},
                                                                                                 {1, 36, 64, 64}}))),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    DeformableConvolutionLayerTest::getTestCaseName);

///* ============= Single Test Case ============= */
const std::vector<std::vector<size_t>> deformable_values_2 = {{1, 54, 28, 28}};
const std::vector<std::vector<size_t>> kernel_shapes_2 = {{1, 3, 3, 3}};
const std::vector<size_t> num_deformable_groups_1 = {3};

const auto deformableConv2DParams_SingleTestCase = ::testing::Combine(
    ::testing::ValuesIn(stride_values),
    ::testing::ValuesIn(pad_begin_values),
    ::testing::ValuesIn(pad_end_values),
    ::testing::ValuesIn(dilation_values),
    ::testing::ValuesIn(num_groups),
    ::testing::ValuesIn(num_deformable_groups_1),
    ::testing::ValuesIn(num_out_channels),
    ::testing::Values(ov::op::PadType::EXPLICIT),
    ::testing::ValuesIn(with_bilinear_interpolation_pad)
);

INSTANTIATE_TEST_SUITE_P(
    smoke_DeformableConvolution2D_SingleTestCase, DeformableConvolutionLayerTest,
    ::testing::Combine(
        deformableConv2DParams_SingleTestCase,
        ::testing::ValuesIn(with_modulated_scalar),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{1, 3, 30, 30},
                                                                                                 {1, 54, 28, 28},
                                                                                                 {1, 3, 3, 3},
                                                                                                 {1, 27, 28, 28}}))),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    DeformableConvolutionLayerTest::getTestCaseName);
/* ============= Multiple groups case ============= */
const auto DeformableConvolution2D_MultipleGroups22 = ::testing::Combine(
    ::testing::ValuesIn(stride_values),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::ValuesIn(dilation_values),
    ::testing::ValuesIn(std::vector<size_t>({2})),
    ::testing::ValuesIn(std::vector<size_t>({2})),
    ::testing::ValuesIn(num_out_channels),
    ::testing::Values(ov::op::PadType::EXPLICIT),
    ::testing::ValuesIn(with_bilinear_interpolation_pad)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_DeformableConvolution2D_MultipleGroups, DeformableConvolutionLayerTest,
        ::testing::Combine(
                DeformableConvolution2D_MultipleGroups22,
                ::testing::ValuesIn(with_modulated_scalar),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(ov::test::static_shapes_to_test_representation(
                        std::vector<ov::Shape>({{1, 4, 3, 3}, {1, 16, 2, 2}, {2, 2, 2, 2}, {1, 8, 2, 2}}))),
                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                DeformableConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_DeformableConvolution2D_MultipleGroups_Batch2, DeformableConvolutionLayerTest,
        ::testing::Combine(
                DeformableConvolution2D_MultipleGroups22,
                ::testing::ValuesIn(with_modulated_scalar),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(ov::test::static_shapes_to_test_representation(
                        std::vector<ov::Shape>({{2, 4, 3, 3}, {2, 16, 2, 2}, {2, 2, 2, 2}, {2, 8, 2, 2}}))),
                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                DeformableConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_DeformableConvolution2D_MultipleGroups_Batch3, DeformableConvolutionLayerTest,
        ::testing::Combine(
                DeformableConvolution2D_MultipleGroups22,
                ::testing::ValuesIn(with_modulated_scalar),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(ov::test::static_shapes_to_test_representation(
                        std::vector<ov::Shape>({{3, 4, 3, 3}, {3, 16, 2, 2}, {2, 2, 2, 2}, {3, 8, 2, 2}}))),
                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                DeformableConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_DeformableConvolution2D_MultipleGroups_Batch4, DeformableConvolutionLayerTest,
        ::testing::Combine(
                DeformableConvolution2D_MultipleGroups22,
                ::testing::ValuesIn(with_modulated_scalar),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(ov::test::static_shapes_to_test_representation(
                        std::vector<ov::Shape>({{4, 4, 3, 3}, {4, 16, 2, 2}, {2, 2, 2, 2}, {4, 8, 2, 2}}))),
                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                DeformableConvolutionLayerTest::getTestCaseName);

const auto DeformableConvolution2D_MultipleGroups_41 = ::testing::Combine(
    ::testing::ValuesIn(stride_values),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::ValuesIn(dilation_values),
    ::testing::ValuesIn(std::vector<size_t>({4})),
    ::testing::ValuesIn(std::vector<size_t>({1})),
    ::testing::ValuesIn(num_out_channels),
    ::testing::Values(ov::op::PadType::EXPLICIT),
    ::testing::ValuesIn(with_bilinear_interpolation_pad)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_DeformableConvolution2D_MultipleGroups_2, DeformableConvolutionLayerTest,
        ::testing::Combine(
                DeformableConvolution2D_MultipleGroups_41,
                ::testing::ValuesIn(with_modulated_scalar),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(ov::test::static_shapes_to_test_representation(
                        std::vector<ov::Shape>({{1, 8, 68, 68}, {1, 18, 66, 66}, {4, 2, 3, 3}, {1, 9, 66, 66}}))),
                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                DeformableConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_DeformableConvolution2D_MultipleGroups_2_Batch2, DeformableConvolutionLayerTest,
        ::testing::Combine(
                DeformableConvolution2D_MultipleGroups_41,
                ::testing::ValuesIn(with_modulated_scalar),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(ov::test::static_shapes_to_test_representation(
                        std::vector<ov::Shape>({{2, 8, 68, 68}, {2, 18, 66, 66}, {4, 2, 3, 3}, {2, 9, 66, 66}}))),
                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                DeformableConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_DeformableConvolution2D_MultipleGroups_2_Batch3, DeformableConvolutionLayerTest,
        ::testing::Combine(
                DeformableConvolution2D_MultipleGroups_41,
                ::testing::ValuesIn(with_modulated_scalar),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(ov::test::static_shapes_to_test_representation(
                        std::vector<ov::Shape>({{3, 8, 68, 68}, {3, 18, 66, 66}, {4, 2, 3, 3}, {3, 9, 66, 66}}))),
                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                DeformableConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_DeformableConvolution2D_MultipleGroups_2_Batch4, DeformableConvolutionLayerTest,
        ::testing::Combine(
                DeformableConvolution2D_MultipleGroups_41,
                ::testing::ValuesIn(with_modulated_scalar),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(ov::test::static_shapes_to_test_representation(
                        std::vector<ov::Shape>({{4, 8, 68, 68}, {4, 18, 66, 66}, {4, 2, 3, 3}, {4, 9, 66, 66}}))),
                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                DeformableConvolutionLayerTest::getTestCaseName);

const auto DeformableConvolution2D_MultipleGroups_42 = ::testing::Combine(
    ::testing::ValuesIn(stride_values),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::ValuesIn(dilation_values),
    ::testing::ValuesIn(std::vector<size_t>({4})),
    ::testing::ValuesIn(std::vector<size_t>({2})),
    ::testing::ValuesIn(num_out_channels),
    ::testing::Values(ov::op::PadType::EXPLICIT),
    ::testing::ValuesIn(with_bilinear_interpolation_pad)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_DeformableConvolution2D_MultipleGroups_3, DeformableConvolutionLayerTest,
        ::testing::Combine(
                DeformableConvolution2D_MultipleGroups_42,
                ::testing::ValuesIn(with_modulated_scalar),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(ov::test::static_shapes_to_test_representation(
                        std::vector<ov::Shape>({{1, 8, 68, 68}, {1, 36, 66, 66}, {4, 2, 3, 3}, {1, 18, 66, 66}}))),
                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                DeformableConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_DeformableConvolution2D_MultipleGroups_3_Batch2, DeformableConvolutionLayerTest,
        ::testing::Combine(
                DeformableConvolution2D_MultipleGroups_42,
                ::testing::ValuesIn(with_modulated_scalar),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(ov::test::static_shapes_to_test_representation(
                        std::vector<ov::Shape>({{2, 8, 68, 68}, {2, 36, 66, 66}, {4, 2, 3, 3}, {2, 18, 66, 66}}))),
                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                DeformableConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_DeformableConvolution2D_MultipleGroups_3_Batch3, DeformableConvolutionLayerTest,
        ::testing::Combine(
                DeformableConvolution2D_MultipleGroups_42,
                ::testing::ValuesIn(with_modulated_scalar),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(ov::test::static_shapes_to_test_representation(
                        std::vector<ov::Shape>({{3, 8, 68, 68}, {3, 36, 66, 66}, {4, 2, 3, 3}, {3, 18, 66, 66}}))),
                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                DeformableConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_DeformableConvolution2D_MultipleGroups_3_Batch4, DeformableConvolutionLayerTest,
        ::testing::Combine(
                DeformableConvolution2D_MultipleGroups_42,
                ::testing::ValuesIn(with_modulated_scalar),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(ov::test::static_shapes_to_test_representation(
                        std::vector<ov::Shape>({{4, 8, 68, 68}, {4, 36, 66, 66}, {4, 2, 3, 3}, {4, 18, 66, 66}}))),
                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                DeformableConvolutionLayerTest::getTestCaseName);
}  // namespace
