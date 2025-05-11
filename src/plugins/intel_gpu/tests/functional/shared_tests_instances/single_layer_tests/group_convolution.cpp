// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/group_convolution.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::GroupConvolutionLayerTest;

const std::vector<ov::element::Type> netPrecisions = {
    ov::element::f32
};

/* ============= 1D GroupConvolution ============= */
// 1D group convolution is not working correctly
const std::vector<std::vector<size_t >> kernels1D = {{3}};
const std::vector<std::vector<size_t >> strides1D = {{1}};
const std::vector<std::vector<ptrdiff_t>> padBegins1D = {{0}};
const std::vector<std::vector<ptrdiff_t>> padEnds1D = {{0}};
const std::vector<std::vector<size_t >> dilations1D = {{1}};
const std::vector<size_t> numOutChannels1D = {16};
const std::vector<size_t> numGroups1D = {2};
const std::vector<size_t> numDWGroups1D = {16};

const auto groupConv1DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels1D),
        ::testing::ValuesIn(strides1D),
        ::testing::ValuesIn(padBegins1D),
        ::testing::ValuesIn(padEnds1D),
        ::testing::ValuesIn(dilations1D),
        ::testing::ValuesIn(numOutChannels1D),
        ::testing::ValuesIn(numGroups1D),
        ::testing::Values(ov::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution1D_ExplicitPadding_Disabled, GroupConvolutionLayerTest,
                        ::testing::Combine(
                                groupConv1DParams_ExplicitPadding,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                        std::vector<std::vector<ov::Shape>>({{{1, 16, 30}}}))),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        GroupConvolutionLayerTest::getTestCaseName);

const auto dwConv1DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels1D),
        ::testing::ValuesIn(strides1D),
        ::testing::ValuesIn(padBegins1D),
        ::testing::ValuesIn(padEnds1D),
        ::testing::ValuesIn(dilations1D),
        ::testing::ValuesIn(numOutChannels1D),
        ::testing::ValuesIn(numDWGroups1D),
        ::testing::Values(ov::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_SUITE_P(smoke_DwGroupConvolution1D_ExplicitPadding, GroupConvolutionLayerTest,
                        ::testing::Combine(
                                dwConv1DParams_ExplicitPadding,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                        std::vector<std::vector<ov::Shape>>({{{1, 16, 30}}}))),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        GroupConvolutionLayerTest::getTestCaseName);

/* ============= 2D GroupConvolution ============= */
const std::vector<std::vector<size_t >> kernels = {{3, 3}};
const std::vector<std::vector<size_t >> strides = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds = {{0, 0}};
const std::vector<std::vector<size_t >> dilations = {{1, 1}};
const std::vector<size_t> numOutChannels = {8, 16};
const std::vector<size_t> numGroups = {2, 8};

const auto groupConv2DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds),
        ::testing::ValuesIn(dilations),
        ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups),
        ::testing::Values(ov::op::PadType::EXPLICIT)
);
const auto groupConv2DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::ValuesIn(dilations),
        ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups),
        ::testing::Values(ov::op::PadType::VALID)
);
const auto groupConv2DParams_AutoPadSameUpper = ::testing::Combine(
        ::testing::Values(std::vector<size_t >({1, 5})),
        ::testing::ValuesIn(strides),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::ValuesIn(dilations),
        ::testing::Values(80),
        ::testing::Values(80),
        ::testing::Values(ov::op::PadType::SAME_UPPER)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution2D_ExplicitPadding, GroupConvolutionLayerTest,
                        ::testing::Combine(
                                groupConv2DParams_ExplicitPadding,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                        std::vector<std::vector<ov::Shape>>({{{1, 16, 30, 30}}}))),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        GroupConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution2D_AutoPadValid, GroupConvolutionLayerTest,
                        ::testing::Combine(
                                groupConv2DParams_AutoPadValid,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                        std::vector<std::vector<ov::Shape>>({{{1, 16, 30, 30}}}))),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        GroupConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution2D_AutoPadSameUpper, GroupConvolutionLayerTest,
                        ::testing::Combine(
                                groupConv2DParams_AutoPadSameUpper,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                        std::vector<std::vector<ov::Shape>>({{{1, 80, 1, 1008}}}))),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        GroupConvolutionLayerTest::getTestCaseName);

/* ============= 3D GroupConvolution ============= */
const std::vector<std::vector<size_t >> kernels3d = {{3, 3, 3}};
const std::vector<std::vector<ptrdiff_t>> paddings3d = {{0, 0, 0}};

const std::vector<std::vector<size_t >> strides3d = {{1, 1, 1}};
const std::vector<std::vector<size_t >> dilations3d = {{1, 1, 1}};

const auto groupConv3DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels3d),
        ::testing::ValuesIn(strides3d),
        ::testing::ValuesIn(paddings3d),
        ::testing::ValuesIn(paddings3d),
        ::testing::ValuesIn(dilations3d),
        ::testing::Values(4),
        ::testing::Values(2),
        ::testing::Values(ov::op::PadType::EXPLICIT)
);
const auto groupConv3DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels3d),
        ::testing::ValuesIn(strides3d),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
        ::testing::ValuesIn(dilations3d),
        ::testing::Values(4),
        ::testing::Values(2),
        ::testing::Values(ov::op::PadType::VALID)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution3D_ExplicitPadding, GroupConvolutionLayerTest,
                        ::testing::Combine(
                                groupConv3DParams_ExplicitPadding,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                        std::vector<std::vector<ov::Shape>>({{{1, 4, 10, 10, 10}}}))),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        GroupConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution3D_AutoPadValid, GroupConvolutionLayerTest,
                        ::testing::Combine(
                                groupConv3DParams_AutoPadValid,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                        std::vector<std::vector<ov::Shape>>({{{1, 4, 10, 10, 10}}}))),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        GroupConvolutionLayerTest::getTestCaseName);

}  // namespace
