// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/deformable_convolution.hpp"
using namespace LayerTestsDefinitions;
namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::I16
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
    ::testing::ValuesIn(deformable_values_0),
    ::testing::ValuesIn(kernel_shapes_0), ::testing::ValuesIn(stride_values),
    ::testing::ValuesIn(pad_begin_values), ::testing::ValuesIn(pad_end_values),
    ::testing::ValuesIn(dilation_values), ::testing::ValuesIn(num_groups),
    ::testing::ValuesIn(num_deformable_groups_0), ::testing::ValuesIn(num_out_channels),
    ::testing::Values(ngraph::op::PadType::EXPLICIT), ::testing::ValuesIn(with_bilinear_interpolation_pad),
    ::testing::ValuesIn(with_modulated_scalar));

const auto deformableConv2DParams_AutoPadValid = ::testing::Combine(
    ::testing::ValuesIn(deformable_values_0),
    ::testing::ValuesIn(kernel_shapes_0), ::testing::ValuesIn(stride_values),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::ValuesIn(dilation_values), ::testing::ValuesIn(num_groups),
    ::testing::ValuesIn(num_deformable_groups_0), ::testing::ValuesIn(num_out_channels),
    ::testing::Values(ngraph::op::PadType::VALID),
    ::testing::ValuesIn(with_bilinear_interpolation_pad),
    ::testing::ValuesIn(with_modulated_scalar));

const auto deformableConv2DParams_DeformableGroups_AutoPadExplicit = ::testing::Combine(
    ::testing::ValuesIn(deformable_values_1),
    ::testing::ValuesIn(kernel_shapes_1), ::testing::ValuesIn(stride_values),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::ValuesIn(dilation_values), ::testing::ValuesIn(num_groups),
    ::testing::ValuesIn(multiple_defor_groups), ::testing::ValuesIn(num_out_channels),
    ::testing::Values(ngraph::op::PadType::EXPLICIT),
    ::testing::ValuesIn(with_bilinear_interpolation_pad),
    ::testing::ValuesIn(with_modulated_scalar));

INSTANTIATE_TEST_SUITE_P(
    smoke_DeformableConvolution2D_ExplicitPadding, DeformableConvolutionLayerTest,
    ::testing::Combine(
        deformableConv2DParams_ExplicitPadding, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 2, 3, 3})),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    DeformableConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_DeformableConvolution2D_AutoPadValid, DeformableConvolutionLayerTest,
    ::testing::Combine(
        deformableConv2DParams_AutoPadValid, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 2, 3, 3})),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    DeformableConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_DeformableConvolution2D_DeformableGroups_ExplicitPadding, DeformableConvolutionLayerTest,
    ::testing::Combine(
        deformableConv2DParams_DeformableGroups_AutoPadExplicit,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 16, 66, 66})),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    DeformableConvolutionLayerTest::getTestCaseName);

///* ============= Single Test Case ============= */
const std::vector<std::vector<size_t>> deformable_values_2 = {{1, 54, 28, 28}};
const std::vector<std::vector<size_t>> kernel_shapes_2 = {{1, 3, 3, 3}};
const std::vector<size_t> num_deformable_groups_1 = {3};

const auto deformableConv2DParams_SingleTestCase = ::testing::Combine(
    ::testing::ValuesIn(deformable_values_2),
    ::testing::ValuesIn(kernel_shapes_2),
    ::testing::ValuesIn(stride_values),
    ::testing::ValuesIn(pad_begin_values),
    ::testing::ValuesIn(pad_end_values),
    ::testing::ValuesIn(dilation_values),
    ::testing::ValuesIn(num_groups),
    ::testing::ValuesIn(num_deformable_groups_1),
    ::testing::ValuesIn(num_out_channels),
    ::testing::Values(ngraph::op::PadType::EXPLICIT),
    ::testing::ValuesIn(with_bilinear_interpolation_pad),
    ::testing::ValuesIn(with_modulated_scalar)
);

INSTANTIATE_TEST_SUITE_P(
    smoke_DeformableConvolution2D_SingleTestCase, DeformableConvolutionLayerTest,
    ::testing::Combine(
        deformableConv2DParams_SingleTestCase,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    DeformableConvolutionLayerTest::getTestCaseName);
/* ============= Multiple groups case ============= */
INSTANTIATE_TEST_SUITE_P(
        smoke_DeformableConvolution2D_MultipleGroups, DeformableConvolutionLayerTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(std::vector<std::vector<size_t>> {{1, 16, 2, 2}}),  // offsets
                        ::testing::ValuesIn(std::vector<std::vector<size_t>> {{2, 2, 2, 2}}),  // ker.
                        ::testing::ValuesIn(stride_values),
                        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                        ::testing::ValuesIn(dilation_values),
                        ::testing::ValuesIn(std::vector<size_t> {2}),  // gr.
                        ::testing::ValuesIn(std::vector<size_t> {2}),  // def. gr.
                        ::testing::ValuesIn(num_out_channels),
                        ::testing::Values(ngraph::op::PadType::EXPLICIT),
                        ::testing::ValuesIn(with_bilinear_interpolation_pad),
                        ::testing::ValuesIn(with_modulated_scalar)),
                        ::testing::ValuesIn(netPrecisions),
                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                        ::testing::Values(InferenceEngine::Layout::ANY),
                        ::testing::Values(InferenceEngine::Layout::ANY),
                        ::testing::Values(std::vector<size_t>({1, 4, 3, 3})),
                        ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        DeformableConvolutionLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(
        smoke_DeformableConvolution2D_MultipleGroups_Batch2, DeformableConvolutionLayerTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(std::vector<std::vector<size_t>> {{2, 16, 2, 2}}),  // offsets
                        ::testing::ValuesIn(std::vector<std::vector<size_t>> {{2, 2, 2, 2}}),  // ker.
                        ::testing::ValuesIn(stride_values),
                        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                        ::testing::ValuesIn(dilation_values),
                        ::testing::ValuesIn(std::vector<size_t> {2}),  // gr.
                        ::testing::ValuesIn(std::vector<size_t> {2}),  // def. gr.
                        ::testing::ValuesIn(num_out_channels),
                        ::testing::Values(ngraph::op::PadType::EXPLICIT),
                        ::testing::ValuesIn(with_bilinear_interpolation_pad),
                        ::testing::ValuesIn(with_modulated_scalar)),
                        ::testing::ValuesIn(netPrecisions),
                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                        ::testing::Values(InferenceEngine::Layout::ANY),
                        ::testing::Values(InferenceEngine::Layout::ANY),
                        ::testing::Values(std::vector<size_t>({2, 4, 3, 3})),
                        ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        DeformableConvolutionLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(
        smoke_DeformableConvolution2D_MultipleGroups_Batch3, DeformableConvolutionLayerTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(std::vector<std::vector<size_t>> {{3, 16, 2, 2}}),  // offsets
                        ::testing::ValuesIn(std::vector<std::vector<size_t>> {{2, 2, 2, 2}}),  // ker.
                        ::testing::ValuesIn(stride_values),
                        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                        ::testing::ValuesIn(dilation_values),
                        ::testing::ValuesIn(std::vector<size_t> {2}),  // gr.
                        ::testing::ValuesIn(std::vector<size_t> {2}),  // def. gr.
                        ::testing::ValuesIn(num_out_channels),
                        ::testing::Values(ngraph::op::PadType::EXPLICIT),
                        ::testing::ValuesIn(with_bilinear_interpolation_pad),
                        ::testing::ValuesIn(with_modulated_scalar)),
                        ::testing::ValuesIn(netPrecisions),
                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                        ::testing::Values(InferenceEngine::Layout::ANY),
                        ::testing::Values(InferenceEngine::Layout::ANY),
                        ::testing::Values(std::vector<size_t>({3, 4, 3, 3})),
                        ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        DeformableConvolutionLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(
        smoke_DeformableConvolution2D_MultipleGroups_Batch4, DeformableConvolutionLayerTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(std::vector<std::vector<size_t>> {{4, 16, 2, 2}}),  // offsets
                        ::testing::ValuesIn(std::vector<std::vector<size_t>> {{2, 2, 2, 2}}),  // ker.
                        ::testing::ValuesIn(stride_values),
                        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                        ::testing::ValuesIn(dilation_values),
                        ::testing::ValuesIn(std::vector<size_t> {2}),  // gr.
                        ::testing::ValuesIn(std::vector<size_t> {2}),  // def. gr.
                        ::testing::ValuesIn(num_out_channels),
                        ::testing::Values(ngraph::op::PadType::EXPLICIT),
                        ::testing::ValuesIn(with_bilinear_interpolation_pad),
                        ::testing::ValuesIn(with_modulated_scalar)),
                        ::testing::ValuesIn(netPrecisions),
                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                        ::testing::Values(InferenceEngine::Layout::ANY),
                        ::testing::Values(InferenceEngine::Layout::ANY),
                        ::testing::Values(std::vector<size_t>({4, 4, 3, 3})),
                        ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        DeformableConvolutionLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(
        smoke_DeformableConvolution2D_MultipleGroups_2, DeformableConvolutionLayerTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(std::vector<std::vector<size_t>> {{1, 18, 66, 66}}),  // offsets
                        ::testing::ValuesIn(std::vector<std::vector<size_t>> {{4, 2, 3, 3}}),  // ker.
                        ::testing::ValuesIn(stride_values),
                        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                        ::testing::ValuesIn(dilation_values),
                        ::testing::ValuesIn(std::vector<size_t> {4}),  // gr.
                        ::testing::ValuesIn(std::vector<size_t> {1}),  // def. gr.
                        ::testing::ValuesIn(num_out_channels),
                        ::testing::Values(ngraph::op::PadType::EXPLICIT),
                        ::testing::ValuesIn(with_bilinear_interpolation_pad),
                        ::testing::ValuesIn(with_modulated_scalar)),
                        ::testing::ValuesIn(netPrecisions),
                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                        ::testing::Values(InferenceEngine::Layout::ANY),
                        ::testing::Values(InferenceEngine::Layout::ANY),
                        ::testing::Values(std::vector<size_t>({1, 8, 68, 68})),
                        ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        DeformableConvolutionLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(
        smoke_DeformableConvolution2D_MultipleGroups_2_Batch2, DeformableConvolutionLayerTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(std::vector<std::vector<size_t>> {{2, 18, 66, 66}}),  // offsets
                        ::testing::ValuesIn(std::vector<std::vector<size_t>> {{4, 2, 3, 3}}),  // ker.
                        ::testing::ValuesIn(stride_values),
                        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                        ::testing::ValuesIn(dilation_values),
                        ::testing::ValuesIn(std::vector<size_t> {4}),  // gr.
                        ::testing::ValuesIn(std::vector<size_t> {1}),  // def. gr.
                        ::testing::ValuesIn(num_out_channels),
                        ::testing::Values(ngraph::op::PadType::EXPLICIT),
                        ::testing::ValuesIn(with_bilinear_interpolation_pad),
                        ::testing::ValuesIn(with_modulated_scalar)),
                        ::testing::ValuesIn(netPrecisions),
                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                        ::testing::Values(InferenceEngine::Layout::ANY),
                        ::testing::Values(InferenceEngine::Layout::ANY),
                        ::testing::Values(std::vector<size_t>({2, 8, 68, 68})),
                        ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        DeformableConvolutionLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(
        smoke_DeformableConvolution2D_MultipleGroups_2_Batch3, DeformableConvolutionLayerTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(std::vector<std::vector<size_t>> {{3, 18, 66, 66}}),  // offsets
                        ::testing::ValuesIn(std::vector<std::vector<size_t>> {{4, 2, 3, 3}}),  // ker.
                        ::testing::ValuesIn(stride_values),
                        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                        ::testing::ValuesIn(dilation_values),
                        ::testing::ValuesIn(std::vector<size_t> {4}),  // gr.
                        ::testing::ValuesIn(std::vector<size_t> {1}),  // def. gr.
                        ::testing::ValuesIn(num_out_channels),
                        ::testing::Values(ngraph::op::PadType::EXPLICIT),
                        ::testing::ValuesIn(with_bilinear_interpolation_pad),
                        ::testing::ValuesIn(with_modulated_scalar)),
                        ::testing::ValuesIn(netPrecisions),
                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                        ::testing::Values(InferenceEngine::Layout::ANY),
                        ::testing::Values(InferenceEngine::Layout::ANY),
                        ::testing::Values(std::vector<size_t>({3, 8, 68, 68})),
                        ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        DeformableConvolutionLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(
        smoke_DeformableConvolution2D_MultipleGroups_2_Batch4, DeformableConvolutionLayerTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(std::vector<std::vector<size_t>> {{4, 18, 66, 66}}),  // offsets
                        ::testing::ValuesIn(std::vector<std::vector<size_t>> {{4, 2, 3, 3}}),  // ker.
                        ::testing::ValuesIn(stride_values),
                        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                        ::testing::ValuesIn(dilation_values),
                        ::testing::ValuesIn(std::vector<size_t> {4}),  // gr.
                        ::testing::ValuesIn(std::vector<size_t> {1}),  // def. gr.
                        ::testing::ValuesIn(num_out_channels),
                        ::testing::Values(ngraph::op::PadType::EXPLICIT),
                        ::testing::ValuesIn(with_bilinear_interpolation_pad),
                        ::testing::ValuesIn(with_modulated_scalar)),
                        ::testing::ValuesIn(netPrecisions),
                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                        ::testing::Values(InferenceEngine::Layout::ANY),
                        ::testing::Values(InferenceEngine::Layout::ANY),
                        ::testing::Values(std::vector<size_t>({4, 8, 68, 68})),
                        ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        DeformableConvolutionLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(
        smoke_DeformableConvolution2D_MultipleGroups_3, DeformableConvolutionLayerTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(std::vector<std::vector<size_t>> {{1, 36, 66, 66}}),  // offsets
                        ::testing::ValuesIn(std::vector<std::vector<size_t>> {{4, 2, 3, 3}}),  // ker.
                        ::testing::ValuesIn(stride_values),
                        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                        ::testing::ValuesIn(dilation_values),
                        ::testing::ValuesIn(std::vector<size_t> {4}),  // gr.
                        ::testing::ValuesIn(std::vector<size_t> {2}),  // def. gr.
                        ::testing::ValuesIn(num_out_channels),
                        ::testing::Values(ngraph::op::PadType::EXPLICIT),
                        ::testing::ValuesIn(with_bilinear_interpolation_pad),
                        ::testing::ValuesIn(with_modulated_scalar)),
                        ::testing::ValuesIn(netPrecisions),
                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                        ::testing::Values(InferenceEngine::Layout::ANY),
                        ::testing::Values(InferenceEngine::Layout::ANY),
                        ::testing::Values(std::vector<size_t>({1, 8, 68, 68})),
                        ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        DeformableConvolutionLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(
        smoke_DeformableConvolution2D_MultipleGroups_3_Batch2, DeformableConvolutionLayerTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(std::vector<std::vector<size_t>> {{2, 36, 66, 66}}),  // offsets
                        ::testing::ValuesIn(std::vector<std::vector<size_t>> {{4, 2, 3, 3}}),  // ker.
                        ::testing::ValuesIn(stride_values),
                        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                        ::testing::ValuesIn(dilation_values),
                        ::testing::ValuesIn(std::vector<size_t> {4}),  // gr.
                        ::testing::ValuesIn(std::vector<size_t> {2}),  // def. gr.
                        ::testing::ValuesIn(num_out_channels),
                        ::testing::Values(ngraph::op::PadType::EXPLICIT),
                        ::testing::ValuesIn(with_bilinear_interpolation_pad),
                        ::testing::ValuesIn(with_modulated_scalar)),
                        ::testing::ValuesIn(netPrecisions),
                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                        ::testing::Values(InferenceEngine::Layout::ANY),
                        ::testing::Values(InferenceEngine::Layout::ANY),
                        ::testing::Values(std::vector<size_t>({2, 8, 68, 68})),
                        ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        DeformableConvolutionLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(
        smoke_DeformableConvolution2D_MultipleGroups_3_Batch3, DeformableConvolutionLayerTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(std::vector<std::vector<size_t>> {{3, 36, 66, 66}}),  // offsets
                        ::testing::ValuesIn(std::vector<std::vector<size_t>> {{4, 2, 3, 3}}),  // ker.
                        ::testing::ValuesIn(stride_values),
                        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                        ::testing::ValuesIn(dilation_values),
                        ::testing::ValuesIn(std::vector<size_t> {4}),  // gr.
                        ::testing::ValuesIn(std::vector<size_t> {2}),  // def. gr.
                        ::testing::ValuesIn(num_out_channels),
                        ::testing::Values(ngraph::op::PadType::EXPLICIT),
                        ::testing::ValuesIn(with_bilinear_interpolation_pad),
                        ::testing::ValuesIn(with_modulated_scalar)),
                        ::testing::ValuesIn(netPrecisions),
                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                        ::testing::Values(InferenceEngine::Layout::ANY),
                        ::testing::Values(InferenceEngine::Layout::ANY),
                        ::testing::Values(std::vector<size_t>({3, 8, 68, 68})),
                        ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        DeformableConvolutionLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(
        smoke_DeformableConvolution2D_MultipleGroups_3_Batch4, DeformableConvolutionLayerTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(std::vector<std::vector<size_t>> {{4, 36, 66, 66}}),  // offsets
                        ::testing::ValuesIn(std::vector<std::vector<size_t>> {{4, 2, 3, 3}}),  // ker.
                        ::testing::ValuesIn(stride_values),
                        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                        ::testing::ValuesIn(dilation_values),
                        ::testing::ValuesIn(std::vector<size_t> {4}),  // gr.
                        ::testing::ValuesIn(std::vector<size_t> {2}),  // def. gr.
                        ::testing::ValuesIn(num_out_channels),
                        ::testing::Values(ngraph::op::PadType::EXPLICIT),
                        ::testing::ValuesIn(with_bilinear_interpolation_pad),
                        ::testing::ValuesIn(with_modulated_scalar)),
                        ::testing::ValuesIn(netPrecisions),
                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                        ::testing::Values(InferenceEngine::Layout::ANY),
                        ::testing::Values(InferenceEngine::Layout::ANY),
                        ::testing::Values(std::vector<size_t>({4, 8, 68, 68})),
                        ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        DeformableConvolutionLayerTest::getTestCaseName);
}  // namespace
