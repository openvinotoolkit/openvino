// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/convolution.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::ConvolutionLayerTest;
const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32,
        ov::element::f16
};

/* ============= 1D Convolution ============= */
const auto conv1DParams = ::testing::Combine(
        ::testing::Values(std::vector<size_t>({1})),
        ::testing::Values(std::vector<size_t>({1})),
        ::testing::Values(std::vector<ptrdiff_t>({0})),
        ::testing::Values(std::vector<ptrdiff_t>({0})),
        ::testing::Values(std::vector<size_t>({1})),
        ::testing::Values(192),
        ::testing::Values(ov::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution1D, ConvolutionLayerTest,
                         ::testing::Combine(
                                 conv1DParams,
                                 ::testing::Values(ov::element::f32),
                                 ::testing::Values(ov::test::static_partial_shapes_to_test_representation(std::vector<ov::PartialShape>({{1, 256, 1}}))),
                                 ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ConvolutionLayerTest::getTestCaseName);

/* ============= 2D Convolution ============= */
const std::vector<std::vector<size_t >> kernels = {{3, 3},
                                                   {3, 5}};
const std::vector<std::vector<size_t >> strides = {{1, 1},
                                                   {1, 3}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{0, 0},
                                                       {0, 3}};
const std::vector<std::vector<ptrdiff_t>> padEnds = {{0, 0},
                                                     {0, 3}};
const std::vector<std::vector<size_t >> dilations = {{1, 1},
                                                     {3, 1}};
const std::vector<size_t> numOutChannels = {1, 5};
const std::vector<ov::op::PadType> padTypes = {
        ov::op::PadType::EXPLICIT,
        ov::op::PadType::VALID
};
const auto conv2DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds),
        ::testing::ValuesIn(dilations),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ov::op::PadType::EXPLICIT)
);
const auto conv2DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::ValuesIn(dilations),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ov::op::PadType::VALID)
);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_ExplicitPadding, ConvolutionLayerTest,
                         ::testing::Combine(
                                 conv2DParams_ExplicitPadding,
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(ov::test::static_shapes_to_test_representation({{1, 3, 30, 30}})),
                                 ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_AutoPadValid, ConvolutionLayerTest,
                         ::testing::Combine(
                                 conv2DParams_AutoPadValid,
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(ov::test::static_shapes_to_test_representation({{1, 3, 30, 30}})),
                                 ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ConvolutionLayerTest::getTestCaseName);
/* ============= 3D Convolution ============= */
const std::vector<std::vector<size_t >> kernels3d = {{3, 3, 3},
                                                     {3, 5, 3}};

const std::vector<std::vector<ptrdiff_t>> paddings3d = {{0, 0, 0},
                                                        {0, 2, 0}};

const std::vector<std::vector<size_t >> strides3d = {{1, 1, 1},
                                                     {1, 2, 1}};

const std::vector<std::vector<size_t >> dilations3d = { {1, 1, 1} };

const std::vector<size_t > numOutChannels3d = {1, 5, 16};

const auto conv3DParams = ::testing::Combine(
        ::testing::ValuesIn(kernels3d),
        ::testing::ValuesIn(strides3d),
        ::testing::ValuesIn(paddings3d),
        ::testing::ValuesIn(paddings3d),
        ::testing::ValuesIn(dilations3d),
        ::testing::ValuesIn(numOutChannels3d),
        ::testing::Values(ov::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution3D_Basic1, ConvolutionLayerTest,
                         ::testing::Combine(
                                 conv3DParams,
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(ov::test::static_shapes_to_test_representation({{1, 3, 10, 10, 10}})),
                                 ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ConvolutionLayerTest::getTestCaseName);
}  // namespace
