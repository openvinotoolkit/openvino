// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/quantized_convolution_backprop_data.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;
using namespace ngraph::helpers;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32
};

const std::vector<size_t> numOutChannels = {16, 32};

const std::vector<size_t > levels = {256};
const std::vector<QuantizationGranularity > granularity = {QuantizationGranularity::Pertensor, QuantizationGranularity::Perchannel};

/* ============= 2D GroupConvolutionBackpropData ============= */
const std::vector<std::vector<size_t >> inputShapes2D = {{1, 16, 10, 10}, {1, 32, 10, 10}};
const std::vector<std::vector<size_t >> kernels2D = {{1, 1}, {3, 3}};
const std::vector<std::vector<size_t >> strides2D = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins2D = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds2D = {{0, 0}};
const std::vector<std::vector<size_t >> dilations2D = {{1, 1}};


const auto quantConvBackpropData2DParams = ::testing::Combine(
        ::testing::ValuesIn(kernels2D),
        ::testing::ValuesIn(strides2D),
        ::testing::ValuesIn(padBegins2D),
        ::testing::ValuesIn(padEnds2D),
        ::testing::ValuesIn(dilations2D),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ngraph::op::PadType::AUTO),
        ::testing::ValuesIn(levels),
        ::testing::ValuesIn(granularity)
);

INSTANTIATE_TEST_SUITE_P(smoke_QuantConvBackpropData2D, QuantConvBackpropDataLayerTest,
                        ::testing::Combine(
                                quantConvBackpropData2DParams,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(inputShapes2D),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        QuantConvBackpropDataLayerTest::getTestCaseName);

/* ============= 3D ConvolutionBackpropData ============= */
const std::vector<std::vector<size_t >> inputShapes3D = {{1, 16, 5, 5, 5}, {1, 32, 5, 5, 5}};
const std::vector<std::vector<size_t >> kernels3D = {{1, 1, 1}, {3, 3, 3}};
const std::vector<std::vector<size_t >> strides3D = {{1, 1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins3D = {{0, 0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds3D = {{0, 0, 0}};
const std::vector<std::vector<size_t >> dilations3D = {{1, 1, 1}};

const auto quantConvBackpropData3DParams = ::testing::Combine(
        ::testing::ValuesIn(kernels3D),
        ::testing::ValuesIn(strides3D),
        ::testing::ValuesIn(padBegins3D),
        ::testing::ValuesIn(padEnds3D),
        ::testing::ValuesIn(dilations3D),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ngraph::op::PadType::AUTO),
        ::testing::ValuesIn(levels),
        ::testing::ValuesIn(granularity)
);

INSTANTIATE_TEST_SUITE_P(smoke_QuantConvBackpropData3D, QuantConvBackpropDataLayerTest,
                        ::testing::Combine(
                                quantConvBackpropData3DParams,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(inputShapes3D),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        QuantConvBackpropDataLayerTest::getTestCaseName);

}  // namespace
