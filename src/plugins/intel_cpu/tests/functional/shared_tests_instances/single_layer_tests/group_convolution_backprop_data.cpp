// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/group_convolution_backprop_data.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::I32};

const std::vector<size_t> numOutChannels = {16, 32};
const std::vector<size_t> numGroups = {2, 8, 16};
const std::vector<std::vector<size_t >> emptyOutputShape = {{}};
const std::vector<std::vector<ptrdiff_t >> emptyOutputPadding = {{}};

/* ============= 1D GroupConvolution ============= */
const std::vector<std::vector<size_t >> inputShapes1D = {{1, 16, 32}};

const std::vector<std::vector<size_t >> kernels1D = {{1}, {3}};
const std::vector<std::vector<size_t>> strides1D = {{1}};
const std::vector<std::vector<ptrdiff_t>> padBegins1D = {{0}};
const std::vector<std::vector<ptrdiff_t>> padEnds1D = {{0}};
const std::vector<std::vector<size_t>> dilations1D = {{1}};

const auto groupConvBackpropData1DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels1D),
        ::testing::ValuesIn(strides1D),
        ::testing::ValuesIn(padBegins1D),
        ::testing::ValuesIn(padEnds1D),
        ::testing::ValuesIn(dilations1D),
        ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups),
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::ValuesIn(emptyOutputPadding)
);

const auto groupConvBackpropData1DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels1D),
        ::testing::ValuesIn(strides1D),
        ::testing::ValuesIn(padBegins1D),
        ::testing::ValuesIn(padEnds1D),
        ::testing::ValuesIn(dilations1D),
        ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups),
        ::testing::Values(ngraph::op::PadType::VALID),
        ::testing::ValuesIn(emptyOutputPadding)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData1D_ExplicitPadding, GroupConvBackpropLayerTest,
                        ::testing::Combine(
                                groupConvBackpropData1DParams_ExplicitPadding,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputShapes1D),
                                ::testing::ValuesIn(emptyOutputShape),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GroupConvBackpropLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData1D_AutoPadValid, GroupConvBackpropLayerTest,
                        ::testing::Combine(
                                groupConvBackpropData1DParams_AutoPadValid,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputShapes1D),
                                ::testing::ValuesIn(emptyOutputShape),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GroupConvBackpropLayerTest::getTestCaseName);

/* ============= 2D GroupConvolution ============= */
const std::vector<std::vector<size_t >> inputShapes2D = {{1, 16, 10, 10},
                                                         {1, 32, 10, 10}};
const std::vector<std::vector<size_t >> kernels2D = {{1, 1}, {3, 3}};
const std::vector<std::vector<size_t >> strides2D = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins2D = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds2D = {{0, 0}};
const std::vector<std::vector<size_t >> dilations2D = {{1, 1}};

const auto groupConvBackpropData2DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels2D),
        ::testing::ValuesIn(strides2D),
        ::testing::ValuesIn(padBegins2D),
        ::testing::ValuesIn(padEnds2D),
        ::testing::ValuesIn(dilations2D),
        ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups),
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::ValuesIn(emptyOutputPadding)
);
const auto groupConvBackpropData2DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels2D),
        ::testing::ValuesIn(strides2D),
        ::testing::ValuesIn(padBegins2D),
        ::testing::ValuesIn(padEnds2D),
        ::testing::ValuesIn(dilations2D),
        ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups),
        ::testing::Values(ngraph::op::PadType::VALID),
        ::testing::ValuesIn(emptyOutputPadding)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData2D_ExplicitPadding, GroupConvBackpropLayerTest,
                        ::testing::Combine(
                                groupConvBackpropData2DParams_ExplicitPadding,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputShapes2D),
                                ::testing::ValuesIn(emptyOutputShape),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GroupConvBackpropLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData2D_AutoPadValid, GroupConvBackpropLayerTest,
                        ::testing::Combine(
                                groupConvBackpropData2DParams_AutoPadValid,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputShapes2D),
                                ::testing::ValuesIn(emptyOutputShape),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GroupConvBackpropLayerTest::getTestCaseName);

const std::vector<std::vector<size_t >> inputShape2D = {{1, 16, 9, 12}};
const std::vector<std::vector<size_t >> outputShapes2D = {{6, 6}, {4, 9}};

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData2D_OutputShapeDefined, GroupConvBackpropLayerTest,
                        ::testing::Combine(
                                groupConvBackpropData2DParams_AutoPadValid,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputShape2D),
                                ::testing::ValuesIn(outputShapes2D),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GroupConvBackpropLayerTest::getTestCaseName);

const std::vector<std::vector<ptrdiff_t>> outputPadding2D = {{1, 1}, {2, 2}};
const std::vector<std::vector<size_t >> testStrides2D = {{3, 3}};

const auto conv2DParams_ExplicitPadding_output_padding = ::testing::Combine(
        ::testing::ValuesIn(kernels2D),
        ::testing::ValuesIn(testStrides2D),
        ::testing::ValuesIn(padBegins2D),
        ::testing::ValuesIn(padEnds2D),
        ::testing::ValuesIn(dilations2D),
        ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups),
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::ValuesIn(outputPadding2D)
);
const auto conv2DParams_AutoPadValid_output_padding = ::testing::Combine(
        ::testing::ValuesIn(kernels2D),
        ::testing::ValuesIn(testStrides2D),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::ValuesIn(dilations2D),
        ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups),
        ::testing::Values(ngraph::op::PadType::VALID),
        ::testing::ValuesIn(outputPadding2D)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData2D_ExplicitPadding_OutputPaddingDefined, GroupConvBackpropLayerTest,
                        ::testing::Combine(
                                conv2DParams_AutoPadValid_output_padding,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputShapes2D),
                                ::testing::ValuesIn(emptyOutputShape),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GroupConvBackpropLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData2D_AutoPadding_OutputPaddingDefined, GroupConvBackpropLayerTest,
                        ::testing::Combine(
                                conv2DParams_ExplicitPadding_output_padding,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputShapes2D),
                                ::testing::ValuesIn(emptyOutputShape),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GroupConvBackpropLayerTest::getTestCaseName);

/* ============= 3D GroupConvolution ============= */
const std::vector<std::vector<size_t >> inputShapes3D = {{1, 16, 5, 5, 5},
                                                         {1, 32, 5, 5, 5}};
const std::vector<std::vector<size_t >> kernels3D = {{1, 1, 1}, {3, 3, 3}};
const std::vector<std::vector<size_t >> strides3D = {{1, 1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins3D = {{0, 0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds3D = {{0, 0, 0}};
const std::vector<std::vector<size_t >> dilations3D = {{1, 1, 1}};

const auto groupConvBackpropData3DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels3D),
        ::testing::ValuesIn(strides3D),
        ::testing::ValuesIn(padBegins3D),
        ::testing::ValuesIn(padEnds3D),
        ::testing::ValuesIn(dilations3D),
        ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups),
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::ValuesIn(emptyOutputPadding)
);
const auto groupConvBackpropData3DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels3D),
        ::testing::ValuesIn(strides3D),
        ::testing::ValuesIn(padBegins3D),
        ::testing::ValuesIn(padEnds3D),
        ::testing::ValuesIn(dilations3D),
        ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups),
        ::testing::Values(ngraph::op::PadType::VALID),
        ::testing::ValuesIn(emptyOutputPadding)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData3D_ExplicitPadding, GroupConvBackpropLayerTest,
                        ::testing::Combine(
                                groupConvBackpropData3DParams_ExplicitPadding,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputShapes3D),
                                ::testing::ValuesIn(emptyOutputShape),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GroupConvBackpropLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData3D_AutoPadValid, GroupConvBackpropLayerTest,
                        ::testing::Combine(
                                groupConvBackpropData3DParams_AutoPadValid,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputShapes3D),
                                ::testing::ValuesIn(emptyOutputShape),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GroupConvBackpropLayerTest::getTestCaseName);

const std::vector<std::vector<size_t >> inputShape3D = {{1, 16, 10, 10, 10}};
const std::vector<std::vector<size_t >> outputShapes3D = {{8, 8, 8}, {10, 10, 10}};

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData3D_OutputShapeDefined, GroupConvBackpropLayerTest,
                        ::testing::Combine(
                                groupConvBackpropData3DParams_AutoPadValid,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputShape3D),
                                ::testing::ValuesIn(outputShapes3D),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GroupConvBackpropLayerTest::getTestCaseName);

const std::vector<std::vector<ptrdiff_t>> outputPadding3D = {{1, 1, 1}, {2, 2, 2}};
const std::vector<std::vector<size_t >> testStrides3D = {{3, 3, 3}};

const auto conv3DParams_ExplicitPadding_output_padding = ::testing::Combine(
        ::testing::ValuesIn(kernels3D),
        ::testing::ValuesIn(testStrides3D),
        ::testing::ValuesIn(padBegins3D),
        ::testing::ValuesIn(padEnds3D),
        ::testing::ValuesIn(dilations3D),
        ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups),
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::ValuesIn(outputPadding3D)
);
const auto conv3DParams_AutoPadValid_output_padding = ::testing::Combine(
        ::testing::ValuesIn(kernels3D),
        ::testing::ValuesIn(testStrides3D),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
        ::testing::ValuesIn(dilations3D),
        ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups),
        ::testing::Values(ngraph::op::PadType::VALID),
        ::testing::ValuesIn(outputPadding3D)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData3D_ExplicitPadding_OutputPaddingDefined, GroupConvBackpropLayerTest,
                        ::testing::Combine(
                                conv3DParams_AutoPadValid_output_padding,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputShapes3D),
                                ::testing::ValuesIn(emptyOutputShape),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GroupConvBackpropLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData3D_AutoPadding_OutputPaddingDefined, GroupConvBackpropLayerTest,
                        ::testing::Combine(
                                conv3DParams_ExplicitPadding_output_padding,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputShapes3D),
                                ::testing::ValuesIn(emptyOutputShape),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GroupConvBackpropLayerTest::getTestCaseName);

}  // namespace
