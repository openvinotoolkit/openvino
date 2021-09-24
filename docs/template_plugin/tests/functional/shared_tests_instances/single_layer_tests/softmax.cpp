// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <utility>

#include "single_layer_tests/softmax.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<InferenceEngine::Layout> inputLayouts2D = {
    InferenceEngine::Layout::NC,
};

const std::vector<std::vector<std::pair<size_t, size_t>>> inputStaticShape2D = {
    {NULL_RANGE}
};

const std::vector<std::vector<std::pair<size_t, size_t>>> inputShape2D = {
    {{1, 200}, {1, 200}}
};

const std::vector<std::vector<InferenceEngine::SizeVector>> targetShapes2D = {
    {InferenceEngine::SizeVector {1, 100}},
    {InferenceEngine::SizeVector {100, 1}},
    {InferenceEngine::SizeVector {10, 10}},
};

const std::vector<size_t> axis2D = {
    0, 1
};

const auto params2D = testing::Combine(
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::ValuesIn(inputLayouts2D),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(inputStaticShape2D),
        testing::ValuesIn(targetShapes2D),
        testing::ValuesIn(axis2D),
        testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
        testing::Values(std::map<std::string, std::string>())
);

const auto params2DDynamicShape = testing::Combine(
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::ValuesIn(inputLayouts2D),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(inputShape2D),
        testing::ValuesIn(targetShapes2D),
        testing::ValuesIn(axis2D),
        testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
        testing::Values(std::map<std::string, std::string>())
);

INSTANTIATE_TEST_SUITE_P(
        smoke_SoftMax2D,
        SoftMaxLayerTest,
        params2D,
        SoftMaxLayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_SoftMax2DDynamicShape,
        SoftMaxLayerTest,
        params2DDynamicShape,
        SoftMaxLayerTest::getTestCaseName
);

const std::vector<std::vector<std::pair<size_t, size_t>>> inputStaticShape4D = {
    {NULL_RANGE}
};

const std::vector<std::vector<std::pair<size_t, size_t>>> inputShape4D = {
    {{1, 200}, {1, 200}, {1, 200}, {1, 200}}
};

const std::vector<std::vector<InferenceEngine::SizeVector>> targetShapes4D = {
    {InferenceEngine::SizeVector {1, 100, 1, 1}},
    {InferenceEngine::SizeVector {1, 3, 4, 3}},
    {InferenceEngine::SizeVector {2, 3, 4, 5}},
};

const std::vector<size_t> axis4D = {0, 1, 2, 3};

const auto params4D = testing::Combine(
    testing::ValuesIn(netPrecisions),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Layout::NCHW),
    testing::Values(InferenceEngine::Layout::ANY),
    testing::ValuesIn(inputStaticShape4D),
    testing::ValuesIn(targetShapes4D),
    testing::ValuesIn(axis4D),
    testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
    testing::Values(std::map<std::string, std::string>())
);

const auto params4DDynamicShape = testing::Combine(
    testing::ValuesIn(netPrecisions),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Layout::NCHW),
    testing::Values(InferenceEngine::Layout::ANY),
    testing::ValuesIn(inputShape4D),
    testing::ValuesIn(targetShapes4D),
    testing::ValuesIn(axis4D),
    testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
    testing::Values(std::map<std::string, std::string>())
);

INSTANTIATE_TEST_SUITE_P(
        smoke_SoftMax4D,
        SoftMaxLayerTest,
        params4D,
        SoftMaxLayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_SoftMax4DDynamicShape,
        SoftMaxLayerTest,
        params4DDynamicShape,
        SoftMaxLayerTest::getTestCaseName
);

}  // namespace
