// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/group_convolution_backprop_data.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::GroupConvBackpropLayerTest;

const std::vector<ov::element::Type> netPrecisions = {ov::element::f32};

const std::vector<ov::Shape> emptyOutputShape = {{}};
const std::vector<std::vector<ptrdiff_t>> emptyOutputPadding = {{}};

const std::vector<size_t> numOutChannels = {16, 32};
const std::vector<size_t> numGroups = {2, 8, 16};

/* ============= 2D GroupConvolution ============= */
const std::vector<std::vector<ov::Shape>> inputShapes2D = {{{1, 16, 10, 10}}, {{1, 32, 10, 10}}};

const std::vector<std::vector<size_t>> kernels2D = {{1, 1}, {3, 3}};
const std::vector<std::vector<size_t>> strides2D = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins2D = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds2D = {{0, 0}};
const std::vector<std::vector<size_t>> dilations2D = {{1, 1}};

const auto groupConvBackpropData2DParams_ExplicitPadding = ::testing::Combine(::testing::ValuesIn(kernels2D),
                                                                              ::testing::ValuesIn(strides2D),
                                                                              ::testing::ValuesIn(padBegins2D),
                                                                              ::testing::ValuesIn(padEnds2D),
                                                                              ::testing::ValuesIn(dilations2D),
                                                                              ::testing::ValuesIn(numOutChannels),
                                                                              ::testing::ValuesIn(numGroups),
                                                                              ::testing::Values(ov::op::PadType::EXPLICIT),
                                                                              ::testing::ValuesIn(emptyOutputPadding));
const auto groupConvBackpropData2DParams_AutoPadValid = ::testing::Combine(::testing::ValuesIn(kernels2D),
                                                                           ::testing::ValuesIn(strides2D),
                                                                           ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                                                                           ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                                                                           ::testing::ValuesIn(dilations2D),
                                                                           ::testing::ValuesIn(numOutChannels),
                                                                           ::testing::ValuesIn(numGroups),
                                                                           ::testing::Values(ov::op::PadType::VALID),
                                                                           ::testing::ValuesIn(emptyOutputPadding));

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData2D_ExplicitPadding,
                         GroupConvBackpropLayerTest,
                         ::testing::Combine(groupConvBackpropData2DParams_ExplicitPadding,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes2D)),
                                            ::testing::ValuesIn(emptyOutputShape),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         GroupConvBackpropLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData2D_AutoPadValid,
                         GroupConvBackpropLayerTest,
                         ::testing::Combine(groupConvBackpropData2DParams_AutoPadValid,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes2D)),
                                            ::testing::ValuesIn(emptyOutputShape),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         GroupConvBackpropLayerTest::getTestCaseName);

/* ============= 3D GroupConvolution ============= */
const std::vector<std::vector<ov::Shape>> inputShapes3D = {{{1, 16, 5, 5, 5}}, {{1, 32, 5, 5, 5}}};

const std::vector<std::vector<size_t>> kernels3D = {{1, 1, 1}, {3, 3, 3}};
const std::vector<std::vector<size_t>> strides3D = {{1, 1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins3D = {{0, 0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds3D = {{0, 0, 0}};
const std::vector<std::vector<size_t>> dilations3D = {{1, 1, 1}};

const auto groupConvBackpropData3DParams_ExplicitPadding = ::testing::Combine(::testing::ValuesIn(kernels3D),
                                                                              ::testing::ValuesIn(strides3D),
                                                                              ::testing::ValuesIn(padBegins3D),
                                                                              ::testing::ValuesIn(padEnds3D),
                                                                              ::testing::ValuesIn(dilations3D),
                                                                              ::testing::ValuesIn(numOutChannels),
                                                                              ::testing::ValuesIn(numGroups),
                                                                              ::testing::Values(ov::op::PadType::EXPLICIT),
                                                                              ::testing::ValuesIn(emptyOutputPadding));
const auto groupConvBackpropData3DParams_AutoPadValid = ::testing::Combine(::testing::ValuesIn(kernels3D),
                                                                           ::testing::ValuesIn(strides3D),
                                                                           ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
                                                                           ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
                                                                           ::testing::ValuesIn(dilations3D),
                                                                           ::testing::ValuesIn(numOutChannels),
                                                                           ::testing::ValuesIn(numGroups),
                                                                           ::testing::Values(ov::op::PadType::VALID),
                                                                           ::testing::ValuesIn(emptyOutputPadding));

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData3D_ExplicitPadding,
                         GroupConvBackpropLayerTest,
                         ::testing::Combine(groupConvBackpropData3DParams_ExplicitPadding,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes3D)),
                                            ::testing::ValuesIn(emptyOutputShape),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         GroupConvBackpropLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData3D_AutoPadValid,
                         GroupConvBackpropLayerTest,
                         ::testing::Combine(groupConvBackpropData3DParams_AutoPadValid,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes3D)),
                                            ::testing::ValuesIn(emptyOutputShape),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         GroupConvBackpropLayerTest::getTestCaseName);
}  // namespace

/* ============= 2D GroupConvolution with OutputShape ============= */
// Define output shapes for 2D GroupConvolution
const std::vector<std::vector<ov::Shape>> outputShapes2D = {{{1, 16, 20, 20}},  // Example of a real outputShape
                                                            {{1, 32, 15, 15}}};

// Define test parameters for 2D GroupConvolution with real outputShape
const auto groupConvBackpropData2DParams_WithOutputShape = ::testing::Combine(::testing::ValuesIn(kernels2D),                // Kernel sizes
                                                                              ::testing::ValuesIn(strides2D),                // Strides
                                                                              ::testing::ValuesIn(padBegins2D),              // Padding at the beginning
                                                                              ::testing::ValuesIn(padEnds2D),                // Padding at the end
                                                                              ::testing::ValuesIn(dilations2D),              // Dilation rates
                                                                              ::testing::ValuesIn(numOutChannels),           // Number of output channels
                                                                              ::testing::ValuesIn(numGroups),                // Number of groups
                                                                              ::testing::Values(ov::op::PadType::EXPLICIT),  // Explicit padding type
                                                                              ::testing::ValuesIn(outputShapes2D)            // Real output shapes
);

// Instantiate test cases for 2D GroupConvolution with real outputShape
INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData2D_WithOutputShape,
                         GroupConvBackpropLayerTest,
                         ::testing::Combine(groupConvBackpropData2DParams_WithOutputShape,
                                            ::testing::ValuesIn(netPrecisions),                                                  // Network precisions
                                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes2D)),  // Input shapes
                                            ::testing::ValuesIn(outputShapes2D),                                                 // Output shapes
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)                                       // Target device
                                            ),
                         GroupConvBackpropLayerTest::getTestCaseName);

/* ============= 3D GroupConvolution with OutputShape ============= */
// Define output shapes for 3D GroupConvolution
const std::vector<std::vector<ov::Shape>> outputShapes3D = {{{1, 16, 10, 10, 10}},  // Example of a real outputShape
                                                            {{1, 32, 8, 8, 8}}};

// Define test parameters for 3D GroupConvolution with real outputShape
const auto groupConvBackpropData3DParams_WithOutputShape = ::testing::Combine(::testing::ValuesIn(kernels3D),                // Kernel sizes
                                                                              ::testing::ValuesIn(strides3D),                // Strides
                                                                              ::testing::ValuesIn(padBegins3D),              // Padding at the beginning
                                                                              ::testing::ValuesIn(padEnds3D),                // Padding at the end
                                                                              ::testing::ValuesIn(dilations3D),              // Dilation rates
                                                                              ::testing::ValuesIn(numOutChannels),           // Number of output channels
                                                                              ::testing::ValuesIn(numGroups),                // Number of groups
                                                                              ::testing::Values(ov::op::PadType::EXPLICIT),  // Explicit padding type
                                                                              ::testing::ValuesIn(outputShapes3D)            // Real output shapes
);

// Instantiate test cases for 3D GroupConvolution with real outputShape
INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData3D_WithOutputShape,
                         GroupConvBackpropLayerTest,
                         ::testing::Combine(groupConvBackpropData3DParams_WithOutputShape,
                                            ::testing::ValuesIn(netPrecisions),                                                  // Network precisions
                                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes3D)),  // Input shapes
                                            ::testing::ValuesIn(outputShapes3D),                                                 // Output shapes
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)                                       // Target device
                                            ),
                         GroupConvBackpropLayerTest::getTestCaseName);