// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/convolution_backprop_data.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::ConvolutionBackpropDataLayerTest;

const std::vector<ov::element::Type> model_type = {
        ov::element::f32,
        ov::element::f16
};

const std::vector<size_t> numOutChannels = {1, 5, 16};
const std::vector<ov::Shape> emptyOutputShape = {{}};
const std::vector<std::vector<ptrdiff_t>> emptyOutputPadding = {{}};

/* ============= 2D ConvolutionBackpropData ============= */
const std::vector<std::vector<ov::Shape>> inputShapes2D_static = {{{1, 3, 30, 30}},
                                                                  {{1, 16, 10, 10}},
                                                                  {{1, 32, 10, 10}}};
const std::vector<std::vector<size_t>> kernels2D = {{1, 1}, {3, 3}, {3, 5}};
const std::vector<std::vector<size_t>> strides2D = {{1, 1}, {1, 3}};
const std::vector<std::vector<ptrdiff_t>> padBegins2D = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds2D = {{0, 0}, {1, 1}};
const std::vector<std::vector<size_t>> dilations2D = {{1, 1}, {2, 2}};

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
                                ::testing::ValuesIn(model_type),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes2D_static)),
                                ::testing::ValuesIn(emptyOutputShape),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        ConvolutionBackpropDataLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData2D_AutoPadValid, ConvolutionBackpropDataLayerTest,
                        ::testing::Combine(
                                conv2DParams_AutoPadValid,
                                ::testing::ValuesIn(model_type),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes2D_static)),
                                ::testing::ValuesIn(emptyOutputShape),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        ConvolutionBackpropDataLayerTest::getTestCaseName);

const std::vector<ov::Shape> inputShape2D_static = {{1, 3, 9, 12}};
const std::vector<ov::Shape> outputShapes2D = {{6, 6}, {4, 9}};

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData2D_OutputShapeDefined, ConvolutionBackpropDataLayerTest,
                        ::testing::Combine(
                                conv2DParams_AutoPadValid,
                                ::testing::ValuesIn(model_type),
                                ::testing::Values(ov::test::static_shapes_to_test_representation(inputShape2D_static)),
                                ::testing::ValuesIn(outputShapes2D),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
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
                                ::testing::ValuesIn(model_type),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes2D_static)),
                                ::testing::ValuesIn(emptyOutputShape),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        ConvolutionBackpropDataLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData2D_AutoPadding_OutputPaddingDefined, ConvolutionBackpropDataLayerTest,
                        ::testing::Combine(
                                conv2DParams_ExplicitPadding_output_padding,
                                ::testing::ValuesIn(model_type),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes2D_static)),
                                ::testing::ValuesIn(emptyOutputShape),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        ConvolutionBackpropDataLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData2D_RoundingOfPadding, ConvolutionBackpropDataLayerTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        ::testing::Values(std::vector<size_t>({31, 1})),
                                        ::testing::Values(std::vector<size_t>({2, 1})),
                                        ::testing::Values(std::vector<ptrdiff_t>({14, 0})),
                                        ::testing::Values(std::vector<ptrdiff_t>({15, 0})),
                                        ::testing::Values(std::vector<size_t>({1, 1})),
                                        ::testing::Values(size_t(4)),
                                        ::testing::Values(ov::op::PadType::SAME_LOWER),
                                        ::testing::Values(std::vector<ptrdiff_t>({0, 0}))),
                                ::testing::Values(ov::element::f32),
                                ::testing::Values(ov::test::static_shapes_to_test_representation({{1, 512, 2, 1}})),
                                ::testing::Values(ov::Shape({ 16, 1 })),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        ConvolutionBackpropDataLayerTest::getTestCaseName);

/* ============= 3D ConvolutionBackpropData ============= */
const std::vector<std::vector<ov::Shape>> inputShapes3D_static = {{{1, 3, 10, 10, 10}},
                                                           {{1, 16, 5, 5, 5}},
                                                           {{1, 32, 5, 5, 5}}};
const std::vector<std::vector<size_t >> kernels3D = {/*{1, 1, 1},*/ {3, 3, 3}};
const std::vector<std::vector<size_t >> strides3D = {{1, 1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins3D = {{0, 0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds3D = {{0, 0, 0}, {1, 1, 1}};
const std::vector<std::vector<size_t >> dilations3D = {{1, 1, 1}, {2, 2, 2}};

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
                                ::testing::ValuesIn(model_type),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes3D_static)),
                                ::testing::ValuesIn(emptyOutputShape),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        ConvolutionBackpropDataLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData3D_AutoPadValid, ConvolutionBackpropDataLayerTest,
                        ::testing::Combine(
                                conv3DParams_AutoPadValid,
                                ::testing::ValuesIn(model_type),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes3D_static)),
                                ::testing::ValuesIn(emptyOutputShape),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        ConvolutionBackpropDataLayerTest::getTestCaseName);

const std::vector<ov::Shape> inputShape3D_static = {{1, 3, 10, 10, 10}};
const std::vector<ov::Shape> outputShapes3D = {{8, 8, 8}, {10, 10, 10}};

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData3D_OutputShapeDefined, ConvolutionBackpropDataLayerTest,
                        ::testing::Combine(
                                conv3DParams_AutoPadValid,
                                ::testing::ValuesIn(model_type),
                                ::testing::Values(ov::test::static_shapes_to_test_representation(inputShape3D_static)),
                                ::testing::ValuesIn(outputShapes3D),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
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
                                ::testing::ValuesIn(model_type),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes3D_static)),
                                ::testing::ValuesIn(emptyOutputShape),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        ConvolutionBackpropDataLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData3D_AutoPadding_OutputPaddingDefined, ConvolutionBackpropDataLayerTest,
                        ::testing::Combine(
                                conv3DParams_ExplicitPadding_output_padding,
                                ::testing::ValuesIn(model_type),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes3D_static)),
                                ::testing::ValuesIn(emptyOutputShape),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        ConvolutionBackpropDataLayerTest::getTestCaseName);

}  // namespace
