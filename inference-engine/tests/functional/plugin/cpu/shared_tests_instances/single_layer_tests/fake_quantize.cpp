// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/fake_quantize.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const ngraph::op::AutoBroadcastSpec numpyBroadcast = ngraph::op::AutoBroadcastType::NUMPY;

const ngraph::op::AutoBroadcastSpec noneBroadcast = ngraph::op::AutoBroadcastType::NONE;

const std::vector<ngraph::op::AutoBroadcastSpec> broadcasts = {
    {ngraph::op::AutoBroadcastType::NUMPY},
    {ngraph::op::AutoBroadcastType::PDPD, -1},
};

const std::vector<InferenceEngine::Precision>
    netPrecisions = {InferenceEngine::Precision::FP32,
                     InferenceEngine::Precision::FP16};

const std::vector<std::vector<size_t>> inputShapes = {{1, 1}, {2, 6}, {1, 1, 1}, {2, 6, 13},
                                                      {1, 1, 1, 1}, {3, 10, 5, 6}, {2, 8, 5, 18}, {2, 16, 3, 18}, {3, 49, 5, 6},
                                                      {1, 1, 1, 1, 1}, {3, 10, 2, 5, 6}, {2, 8, 1, 5, 18}, {2, 16, 4, 3, 18}, {3, 49, 7, 5, 6}};
const std::vector<std::vector<size_t>> constShapes = {{1}};
const std::vector<size_t> levels = {16, 255, 256};

const std::pair<std::string, std::map<std::string, std::string>> config = {};
const std::vector<float> fqArgs = {};
const std::vector<float> inputParams = {};

const auto fqParams = ::testing::Combine(
        ::testing::ValuesIn(levels),
        ::testing::ValuesIn(constShapes),
        ::testing::Values(fqArgs),
        ::testing::Values(inputParams),
        ::testing::ValuesIn(broadcasts)
);

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantize, FakeQuantizeLayerTestRevise,
                        ::testing::Combine(
                                fqParams,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputShapes),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(config)),
                        FakeQuantizeLayerTestRevise::getTestCaseName);


const std::vector<size_t> singleShape = {3, 4, 2, 5};
const auto noneBroadcastFqParams = ::testing::Combine(
        ::testing::ValuesIn(levels),
        ::testing::Values(singleShape),
        ::testing::Values(fqArgs),
        ::testing::Values(inputParams),
        ::testing::Values(noneBroadcast)
);

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantizeNoneBroadcast, FakeQuantizeLayerTestRevise,
                        ::testing::Combine(
                                noneBroadcastFqParams,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(singleShape),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(config)),
                        FakeQuantizeLayerTestRevise::getTestCaseName);

const std::vector<std::vector<size_t>> inputShapesPerChannel = {{11, 10, 22, 19}, {11, 10, 5, 6}};
const std::vector<std::vector<size_t>> constShapesPerChannelAxis0 = {{11, 1, 1, 1}};
const std::vector<std::vector<size_t>> constShapesPerChannelAxis1 = {{1, 10, 1, 1}, {10, 1, 1}};

const auto fqParamsPerChannelAxis0 = ::testing::Combine(
        ::testing::ValuesIn(levels),
        ::testing::ValuesIn(constShapesPerChannelAxis0),
        ::testing::Values(fqArgs),
        ::testing::Values(inputParams),
        ::testing::Values(numpyBroadcast)
);

const auto fqParamsPerChannelAxis1 = ::testing::Combine(
        ::testing::ValuesIn(levels),
        ::testing::ValuesIn(constShapesPerChannelAxis1),
        ::testing::Values(fqArgs),
        ::testing::Values(inputParams),
        ::testing::Values(numpyBroadcast)
);

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantizePerChannelAxis0, FakeQuantizeLayerTestRevise,
                        ::testing::Combine(
                                fqParamsPerChannelAxis0,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputShapesPerChannel),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(config)),
                        FakeQuantizeLayerTestRevise::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantizePerChannelAxis1, FakeQuantizeLayerTestRevise,
                        ::testing::Combine(
                                fqParamsPerChannelAxis1,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputShapesPerChannel),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(config)),
                        FakeQuantizeLayerTestRevise::getTestCaseName);

const std::vector<std::vector<size_t>> inputShapesPerChannel2D = {{1, 10}};
const std::vector<std::vector<size_t>> constShapesPerChannel2D = { {10}, {1, 10}, {1} };
const auto fqParamsPerChannel2D = ::testing::Combine(
    ::testing::ValuesIn(levels),
    ::testing::ValuesIn(constShapesPerChannel2D),
    ::testing::Values(fqArgs),
    ::testing::Values(inputParams),
    ::testing::Values(numpyBroadcast)
);

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantizePerChannel2D, FakeQuantizeLayerTestRevise,
    ::testing::Combine(
        fqParamsPerChannel2D,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(inputShapesPerChannel2D),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(config)),
    FakeQuantizeLayerTestRevise::getTestCaseName);

}  // namespace
