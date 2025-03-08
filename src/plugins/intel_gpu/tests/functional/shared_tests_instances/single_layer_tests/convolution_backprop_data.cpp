// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/convolution_backprop_data.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::ConvolutionBackpropDataLayerTest;
using ov::test::convBackpropDataLayerTestParamsSet;

const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32,
        ov::element::f16
};

const std::vector<size_t> numOutChannels = {1, 5, 16};
const std::vector<ov::Shape> emptyOutputShape = {{}};
const std::vector<std::vector<ptrdiff_t>> emptyOutputPadding = {{}};

/* ============= 2D ConvolutionBackpropData ============= */
const std::vector<ov::element::Type> netPrecisions2D = {
        ov::element::f32,
        ov::element::f16
};

const std::vector<std::vector<ov::Shape>> inputShapes2D = {{{1, 3, 30, 30}},
                                                           {{1, 16, 10, 10}},
                                                           {{1, 32, 10, 10}}};
const std::vector<std::vector<size_t >> kernels2D = {{1, 1}, {3, 3}, {3, 5}};
const std::vector<std::vector<size_t >> strides2D = {{1, 3}};
const std::vector<std::vector<ptrdiff_t>> padBegins2D = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds2D = {{0, 0}, {1, 1}};
const std::vector<std::vector<size_t >> dilations2D = {{1, 1}};

const auto conv2DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels2D),
        ::testing::ValuesIn(strides2D),
        ::testing::ValuesIn(padBegins2D),
        ::testing::ValuesIn(padEnds2D),
        ::testing::ValuesIn(dilations2D),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ov::op::PadType::EXPLICIT),
        ::testing::ValuesIn(emptyOutputPadding)
);
const auto conv2DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels2D),
        ::testing::ValuesIn(strides2D),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::ValuesIn(dilations2D),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ov::op::PadType::VALID),
        ::testing::ValuesIn(emptyOutputPadding)
);

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData2D_ExplicitPadding, ConvolutionBackpropDataLayerTest,
                        ::testing::Combine(
                                conv2DParams_ExplicitPadding,
                                ::testing::ValuesIn(netPrecisions2D),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes2D)),
                                ::testing::ValuesIn(emptyOutputShape),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        ConvolutionBackpropDataLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData2D_AutoPadValid, ConvolutionBackpropDataLayerTest,
                        ::testing::Combine(
                                conv2DParams_AutoPadValid,
                                ::testing::ValuesIn(netPrecisions2D),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes2D)),
                                ::testing::ValuesIn(emptyOutputShape),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        ConvolutionBackpropDataLayerTest::getTestCaseName);

const std::vector<std::vector<ptrdiff_t>> outputPadding2D = {{1, 1}, {2, 2}};
const std::vector<std::vector<size_t >> testStrides2D = {{3, 3}};

const auto conv2DParams_ExplicitPadding_output_padding = ::testing::Combine(
        ::testing::ValuesIn(kernels2D),
        ::testing::ValuesIn(testStrides2D),
        ::testing::ValuesIn(padBegins2D),
        ::testing::ValuesIn(padEnds2D),
        ::testing::ValuesIn(dilations2D),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ov::op::PadType::EXPLICIT),
        ::testing::ValuesIn(outputPadding2D)
);
const auto conv2DParams_AutoPadValid_output_padding = ::testing::Combine(
        ::testing::ValuesIn(kernels2D),
        ::testing::ValuesIn(testStrides2D),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::ValuesIn(dilations2D),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ov::op::PadType::VALID),
        ::testing::ValuesIn(outputPadding2D)
);

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData2D_ExplicitPadding_OutputPaddingDefined, ConvolutionBackpropDataLayerTest,
                        ::testing::Combine(
                                conv2DParams_AutoPadValid_output_padding,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes2D)),
                                ::testing::ValuesIn(emptyOutputShape),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        ConvolutionBackpropDataLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData2D_AutoPadding_OutputPaddingDefined, ConvolutionBackpropDataLayerTest,
                        ::testing::Combine(
                                conv2DParams_ExplicitPadding_output_padding,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes2D)),
                                ::testing::ValuesIn(emptyOutputShape),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        ConvolutionBackpropDataLayerTest::getTestCaseName);

/* ============= 3D ConvolutionBackpropData ============= */

const std::vector<ov::element::Type> netPrecisions3D = {
        ov::element::f32,
};
const std::vector<std::vector<ov::Shape>> inputShapes3D = {{{1, 3, 10, 10, 10}},
                                                           {{1, 16, 5, 5, 5}},
                                                           {{1, 32, 5, 5, 5}}};
const std::vector<std::vector<size_t>> kernels3D = {/*{1, 1, 1}, */{3, 3, 3}};
const std::vector<std::vector<size_t>> strides3D = {{1, 1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins3D = {{0, 0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds3D = {{0, 0, 0}, {1, 1, 1}};
const std::vector<std::vector<size_t>> dilations3D = {{1, 1, 1}};

const auto conv3DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels3D),
        ::testing::ValuesIn(strides3D),
        ::testing::ValuesIn(padBegins3D),
        ::testing::ValuesIn(padEnds3D),
        ::testing::ValuesIn(dilations3D),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ov::op::PadType::EXPLICIT),
        ::testing::ValuesIn(emptyOutputPadding)
);
const auto conv3DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels3D),
        ::testing::ValuesIn(strides3D),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
        ::testing::ValuesIn(dilations3D),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ov::op::PadType::VALID),
        ::testing::ValuesIn(emptyOutputPadding)
);

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData3D_ExplicitPadding, ConvolutionBackpropDataLayerTest,
                        ::testing::Combine(
                                conv3DParams_ExplicitPadding,
                                ::testing::ValuesIn(netPrecisions3D),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes3D)),
                                ::testing::ValuesIn(emptyOutputShape),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        ConvolutionBackpropDataLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData3D_AutoPadValid, ConvolutionBackpropDataLayerTest,
                        ::testing::Combine(
                                conv3DParams_AutoPadValid,
                                ::testing::ValuesIn(netPrecisions3D),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes3D)),
                                ::testing::ValuesIn(emptyOutputShape),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        ConvolutionBackpropDataLayerTest::getTestCaseName);

const std::vector<std::vector<ptrdiff_t>> outputPadding3D = {{1, 1, 1}, {2, 2, 2}};
const std::vector<std::vector<size_t >> testStrides3D = {{3, 3, 3}};

const auto conv3DParams_ExplicitPadding_output_padding = ::testing::Combine(
        ::testing::ValuesIn(kernels3D),
        ::testing::ValuesIn(testStrides3D),
        ::testing::ValuesIn(padBegins3D),
        ::testing::ValuesIn(padEnds3D),
        ::testing::ValuesIn(dilations3D),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ov::op::PadType::EXPLICIT),
        ::testing::ValuesIn(outputPadding3D)
);
const auto conv3DParams_AutoPadValid_output_padding = ::testing::Combine(
        ::testing::ValuesIn(kernels3D),
        ::testing::ValuesIn(testStrides3D),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
        ::testing::ValuesIn(dilations3D),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ov::op::PadType::VALID),
        ::testing::ValuesIn(outputPadding3D)
);

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData3D_ExplicitPadding_OutputPaddingDefined, ConvolutionBackpropDataLayerTest,
                        ::testing::Combine(
                                conv3DParams_AutoPadValid_output_padding,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes3D)),
                                ::testing::ValuesIn(emptyOutputShape),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        ConvolutionBackpropDataLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData3D_AutoPadding_OutputPaddingDefined, ConvolutionBackpropDataLayerTest,
                        ::testing::Combine(
                                conv3DParams_ExplicitPadding_output_padding,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes3D)),
                                ::testing::ValuesIn(emptyOutputShape),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        ConvolutionBackpropDataLayerTest::getTestCaseName);

const std::vector<size_t> numOutChannels1d = {256};

/* ============= 1D ConvolutionBackpropData ============= */
const std::vector<ov::element::Type> netPrecisions1D = {
        ov::element::f32
};

const std::vector<std::vector<ov::Shape>> inputShapes1D = {{{1, 512, 577}}};
const std::vector<std::vector<size_t >> kernels1D = {{16}};
const std::vector<std::vector<size_t >> strides1D = {{8}};
const std::vector<std::vector<ptrdiff_t>> padBegins1D = {{4}};
const std::vector<std::vector<ptrdiff_t>> padEnds1D = {{4}};
const std::vector<std::vector<size_t >> dilations1D = {{1}};


const std::vector<std::vector<ptrdiff_t>> outputPadding1D = {{0}};

const auto conv1DParams_ExplicitPadding_output_padding = ::testing::Combine(
        ::testing::ValuesIn(kernels1D),
        ::testing::ValuesIn(strides1D),
        ::testing::ValuesIn(padBegins1D),
        ::testing::ValuesIn(padEnds1D),
        ::testing::ValuesIn(dilations1D),
        ::testing::ValuesIn(numOutChannels1d),
        ::testing::Values(ov::op::PadType::EXPLICIT),
        ::testing::ValuesIn(outputPadding1D)
);

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData1D_ExplicitPadding, ConvolutionBackpropDataLayerTest,
                        ::testing::Combine(
                                conv1DParams_ExplicitPadding_output_padding,
                                ::testing::ValuesIn(netPrecisions1D),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes1D)),
                                ::testing::ValuesIn(emptyOutputShape),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        ConvolutionBackpropDataLayerTest::getTestCaseName);
}  // namespace
