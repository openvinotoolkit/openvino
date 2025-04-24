// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/quantized_group_convolution.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> netPrecisions = {ov::element::f32};

const std::vector<size_t> numOutChannels = {3, 24, 48};
const std::vector<size_t> numGroups = {3};

const std::vector<size_t> levels = {256};
const std::vector<ov::test::utils::QuantizationGranularity> granularity = {
    ov::test::utils::QuantizationGranularity::Pertensor,
    ov::test::utils::QuantizationGranularity::Perchannel};
const std::vector<bool> quantizeWeights = {false, true};

/* ============= 2D GroupConvolution ============= */
const std::vector<ov::Shape> inputShapes2D = {{1, 3, 10, 10}, {1, 24, 10, 10}};
const std::vector<ov::Shape> kernels2D = {{1, 1}, {3, 3}};
const std::vector<ov::Shape> strides2D = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins2D = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds2D = {{0, 0}};
const std::vector<ov::Shape> dilations2D = {{1, 1}};


const auto quantGroupConv2DParams = ::testing::Combine(
        ::testing::ValuesIn(kernels2D),
        ::testing::ValuesIn(strides2D),
        ::testing::ValuesIn(padBegins2D),
        ::testing::ValuesIn(padEnds2D),
        ::testing::ValuesIn(dilations2D),
        ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups),
        ::testing::ValuesIn(levels),
        ::testing::ValuesIn(granularity),
        ::testing::ValuesIn(quantizeWeights)
);

INSTANTIATE_TEST_SUITE_P(smoke_QuantGroupConv2D, QuantGroupConvLayerTest,
                        ::testing::Combine(
                                quantGroupConv2DParams,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(inputShapes2D),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        QuantGroupConvLayerTest::getTestCaseName);

/* ============= 3D GroupConvolution ============= */
const std::vector<ov::Shape> inputShapes3D = {{1, 3, 5, 5, 5}, {1, 24, 5, 5, 5}};
const std::vector<ov::Shape> kernels3D = {{3, 3, 3}};
const std::vector<ov::Shape> strides3D = {{1, 1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins3D = {{0, 0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds3D = {{0, 0, 0}};
const std::vector<ov::Shape> dilations3D = {{1, 1, 1}};

const auto quantGroupConv3DParams = ::testing::Combine(
        ::testing::ValuesIn(kernels3D),
        ::testing::ValuesIn(strides3D),
        ::testing::ValuesIn(padBegins3D),
        ::testing::ValuesIn(padEnds3D),
        ::testing::ValuesIn(dilations3D),
        ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups),
        ::testing::ValuesIn(levels),
        ::testing::ValuesIn(granularity),
        ::testing::ValuesIn(quantizeWeights)
);

INSTANTIATE_TEST_SUITE_P(smoke_QuantGroupConv3D, QuantGroupConvLayerTest,
                        ::testing::Combine(
                                quantGroupConv3DParams,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(inputShapes3D),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        QuantGroupConvLayerTest::getTestCaseName);

}  // namespace
