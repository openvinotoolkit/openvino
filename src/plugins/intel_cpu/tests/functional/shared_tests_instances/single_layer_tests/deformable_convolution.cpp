// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "common_test_utils/test_constants.hpp"
#include "single_op_tests/deformable_convolution.hpp"

namespace {
using ov::test::DeformableConvolutionLayerTest;

const std::vector<ov::element::Type> netPrecisions = {
    ov::element::f32, ov::element::f16,
    ov::element::i32, ov::element::i16};

/* ============= 2D DeformableConvolution ============= */
const std::vector<std::vector<size_t>> deformable_vals = {{1, 16, 2, 2}};
const std::vector<std::vector<size_t>> kernels = {{2, 2, 2, 2}};
const std::vector<std::vector<size_t>> strides = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds ={{0, 0}};
const std::vector<std::vector<size_t>> dilations = {{1, 1}};
const std::vector<size_t> groups = {1};
const std::vector<size_t> defor_groups = {2};
const std::vector<size_t> numOutChannels = {1, 5};
const std::vector<size_t> multiple_defor_groups = {4};
const std::vector<std::vector<size_t>> deform_vals = {{1, 72, 64, 64}};
const std::vector<std::vector<size_t>> kernel = {{16, 16, 3, 3}};

const std::vector<ov::Shape> shapes_no_modulation {
    {1, 2, 3, 3},
    {1, 16, 2, 2}, //deformable_vals
    {2, 2, 2, 2},  //kernels
};

const std::vector<ov::Shape> shapes_with_modulation {
    {1, 2, 3, 3},
    {1, 16, 2, 2}, //deformable_vals
    {2, 2, 2, 2},  //kernels
    {1, 8, 2, 2},  //modulation_shape
};

const std::vector<bool> with_bilinear_interpolation_pad = { false, true };
const std::vector<bool> with_modulated_scalar = { false, true };

const auto deformableConv2DParams_ExplicitPadding = ::testing::Combine(
    ::testing::ValuesIn(strides),
    ::testing::ValuesIn(padBegins), ::testing::ValuesIn(padEnds),
    ::testing::ValuesIn(dilations), ::testing::ValuesIn(groups),
    ::testing::ValuesIn(defor_groups), ::testing::ValuesIn(numOutChannels),
    ::testing::Values(ov::op::PadType::EXPLICIT), ::testing::ValuesIn(with_bilinear_interpolation_pad));

const auto deformableConv2DParams_AutoPadValid = ::testing::Combine(
    ::testing::ValuesIn(strides),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::ValuesIn(dilations), ::testing::ValuesIn(groups),
    ::testing::ValuesIn(defor_groups), ::testing::ValuesIn(numOutChannels),
    ::testing::Values(ov::op::PadType::VALID),
    ::testing::ValuesIn(with_bilinear_interpolation_pad));

const auto deformableConv2DParams_DeformableGroups_AutoPadExplicit = ::testing::Combine(
    ::testing::ValuesIn(strides),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::ValuesIn(dilations), ::testing::ValuesIn(groups),
    ::testing::ValuesIn(multiple_defor_groups), ::testing::ValuesIn(numOutChannels),
    ::testing::Values(ov::op::PadType::EXPLICIT),
    ::testing::ValuesIn(with_bilinear_interpolation_pad));

INSTANTIATE_TEST_SUITE_P(
    smoke_DeformableConvolution2D_ExplicitPadding_NoModulation, DeformableConvolutionLayerTest,
    ::testing::Combine(
        deformableConv2DParams_ExplicitPadding,
        ::testing::Values(false),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::test::static_shapes_to_test_representation(shapes_no_modulation)),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
    DeformableConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_DeformableConvolution2D_ExplicitPadding_WithModulation, DeformableConvolutionLayerTest,
    ::testing::Combine(
        deformableConv2DParams_ExplicitPadding,
        ::testing::Values(true),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::test::static_shapes_to_test_representation(shapes_with_modulation)),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
    DeformableConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_DeformableConvolution2D_AutoPadValid_NoModulation, DeformableConvolutionLayerTest,
    ::testing::Combine(
        deformableConv2DParams_AutoPadValid,
        ::testing::Values(false),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::test::static_shapes_to_test_representation(shapes_no_modulation)),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
    DeformableConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_DeformableConvolution2D_AutoPadValid_WithModulation, DeformableConvolutionLayerTest,
    ::testing::Combine(
        deformableConv2DParams_AutoPadValid,
        ::testing::Values(true),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::test::static_shapes_to_test_representation(shapes_with_modulation)),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
    DeformableConvolutionLayerTest::getTestCaseName);

const std::vector<ov::Shape> shapes_2d_deformable_groups_no_modulation {
    {1, 16, 66, 66},
    {1, 72, 64, 64}, //deformable_vals
    {16, 16, 3, 3},  //kernels
};

const std::vector<ov::Shape> shapes_2d_deformable_groups_with_modulation {
    {1, 16, 66, 66},
    {1, 72, 64, 64}, //deformable_vals
    {16, 16, 3, 3},  //kernels
    {1, 36, 64, 64},  //modulation_shape
};

INSTANTIATE_TEST_SUITE_P(
    smoke_DeformableConvolution2D_DeformableGroups_ExplicitPadding_NoModulation, DeformableConvolutionLayerTest,
    ::testing::Combine(
        deformableConv2DParams_DeformableGroups_AutoPadExplicit,
        ::testing::Values(false),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::test::static_shapes_to_test_representation(shapes_2d_deformable_groups_no_modulation)),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
    DeformableConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_DeformableConvolution2D_DeformableGroups_ExplicitPadding_WithModulation, DeformableConvolutionLayerTest,
    ::testing::Combine(
        deformableConv2DParams_DeformableGroups_AutoPadExplicit,
        ::testing::Values(true),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::test::static_shapes_to_test_representation(shapes_2d_deformable_groups_with_modulation)),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
    DeformableConvolutionLayerTest::getTestCaseName);

/* ============= Single Test Case ============= */
const std::vector<std::vector<size_t>> single_deform_vals = {{1, 54, 28, 28}};
const std::vector<std::vector<size_t>> single_kernel = {{1, 3, 3, 3}};
const std::vector<size_t> single_deform_groups = {3};

const std::vector<ov::Shape> shapes_single_no_modulation {
    {1, 3, 30, 30},
    {1, 54, 28, 28}, //deformable_vals
    {1, 3, 3, 3},  //kernels
};

const std::vector<ov::Shape> shapes_single_with_modulation {
    {1, 3, 30, 30},
    {1, 54, 28, 28}, //deformable_vals
    {1, 3, 3, 3},  //kernels
    {1, 27, 28, 28},  //modulation_shape
};


const auto deformableConv2DParams_SingleTestCase = ::testing::Combine(
    ::testing::ValuesIn(strides),
    ::testing::ValuesIn(padBegins),
    ::testing::ValuesIn(padEnds),
    ::testing::ValuesIn(dilations),
    ::testing::ValuesIn(groups),
    ::testing::ValuesIn(single_deform_groups),
    ::testing::ValuesIn(numOutChannels),
    ::testing::Values(ov::op::PadType::EXPLICIT),
    ::testing::ValuesIn(with_bilinear_interpolation_pad));

INSTANTIATE_TEST_SUITE_P(
    smoke_DeformableConvolution2D_SingleTestCase_NoModulation, DeformableConvolutionLayerTest,
    ::testing::Combine(
        deformableConv2DParams_SingleTestCase,
        ::testing::Values(false),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::test::static_shapes_to_test_representation(shapes_single_no_modulation)),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
    DeformableConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_DeformableConvolution2D_SingleTestCase_WithModulation, DeformableConvolutionLayerTest,
    ::testing::Combine(
        deformableConv2DParams_SingleTestCase,
        ::testing::Values(true),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::test::static_shapes_to_test_representation(shapes_single_with_modulation)),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
    DeformableConvolutionLayerTest::getTestCaseName);

/* ============= Multiple groups case ============= */

const std::vector<ov::Shape> shapes_multiple_groups_no_modulation {
    {1, 4, 3, 3},
    {1, 16, 2, 2}, //deformable_vals
    {2, 2, 2, 2},  //kernels
};

const std::vector<ov::Shape> shapes_multiple_groups_with_modulation {
    {1, 4, 3, 3},
    {1, 16, 2, 2}, //deformable_vals
    {2, 2, 2, 2},  //kernels
    {1, 8, 2, 2},  //modulation_shape
};

INSTANTIATE_TEST_SUITE_P(
        smoke_DeformableConvolution2D_MultipleGroups_NoModulation, DeformableConvolutionLayerTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(strides),
                        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                        ::testing::ValuesIn(dilations),
                        ::testing::ValuesIn(std::vector<size_t> {2}),  // gr.
                        ::testing::ValuesIn(std::vector<size_t> {2}),  // def. gr.
                        ::testing::ValuesIn(numOutChannels),
                        ::testing::Values(ov::op::PadType::EXPLICIT),
                        ::testing::ValuesIn(with_bilinear_interpolation_pad)),
                        ::testing::Values(false),
                        ::testing::ValuesIn(netPrecisions),
                        ::testing::Values(ov::test::static_shapes_to_test_representation(shapes_multiple_groups_no_modulation)),
                        ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        DeformableConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_DeformableConvolution2D_MultipleGroups_WithModulation, DeformableConvolutionLayerTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(strides),
                        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                        ::testing::ValuesIn(dilations),
                        ::testing::ValuesIn(std::vector<size_t> {2}),  // gr.
                        ::testing::ValuesIn(std::vector<size_t> {2}),  // def. gr.
                        ::testing::ValuesIn(numOutChannels),
                        ::testing::Values(ov::op::PadType::EXPLICIT),
                        ::testing::ValuesIn(with_bilinear_interpolation_pad)),
                        ::testing::Values(true),
                        ::testing::ValuesIn(netPrecisions),
                        ::testing::Values(ov::test::static_shapes_to_test_representation(shapes_multiple_groups_with_modulation)),
                        ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        DeformableConvolutionLayerTest::getTestCaseName);

const std::vector<ov::Shape> shapes_multiple_groups_2_no_modulation {
    {1, 8, 68, 68},
    {1, 18, 66, 66}, //deformable_vals
    {4, 2, 3, 3},  //kernels
};

const std::vector<ov::Shape> shapes_multiple_groups_2_with_modulation {
    {1, 8, 68, 68},
    {1, 18, 66, 66}, //deformable_vals
    {4, 2, 3, 3},  //kernels
    {1, 9, 66, 66},  //modulation_shape
};

INSTANTIATE_TEST_SUITE_P(
        smoke_DeformableConvolution2D_MultipleGroups_2_NoModulation, DeformableConvolutionLayerTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(strides),
                        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                        ::testing::ValuesIn(dilations),
                        ::testing::ValuesIn(std::vector<size_t> {4}),  // gr.
                        ::testing::ValuesIn(std::vector<size_t> {1}),  // def. gr.
                        ::testing::ValuesIn(numOutChannels),
                        ::testing::Values(ov::op::PadType::EXPLICIT),
                        ::testing::ValuesIn(with_bilinear_interpolation_pad)),
                        ::testing::Values(false),
                        ::testing::ValuesIn(netPrecisions),
                        ::testing::Values(ov::test::static_shapes_to_test_representation(shapes_multiple_groups_2_no_modulation)),
                        ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        DeformableConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_DeformableConvolution2D_MultipleGroups_2_WithModulation, DeformableConvolutionLayerTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(strides),
                        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                        ::testing::ValuesIn(dilations),
                        ::testing::ValuesIn(std::vector<size_t> {4}),  // gr.
                        ::testing::ValuesIn(std::vector<size_t> {1}),  // def. gr.
                        ::testing::ValuesIn(numOutChannels),
                        ::testing::Values(ov::op::PadType::EXPLICIT),
                        ::testing::ValuesIn(with_bilinear_interpolation_pad)),
                        ::testing::Values(true),
                        ::testing::ValuesIn(netPrecisions),
                        ::testing::Values(ov::test::static_shapes_to_test_representation(shapes_multiple_groups_2_with_modulation)),
                        ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        DeformableConvolutionLayerTest::getTestCaseName);
}  // namespace
