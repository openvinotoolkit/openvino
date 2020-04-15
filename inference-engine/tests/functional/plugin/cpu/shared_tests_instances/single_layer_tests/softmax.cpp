// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/softmax.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<InferenceEngine::Precision> inputPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<InferenceEngine::Layout> inputLayouts2D = {
    InferenceEngine::Layout::NC,
};

const std::vector<InferenceEngine::SizeVector> inputShapes2D = {
    InferenceEngine::SizeVector {1, 100},
};

const std::vector<size_t> axis2D = {
    1
};

const auto params2D = testing::Combine(
    testing::ValuesIn(netPrecisions),
    testing::ValuesIn(inputPrecisions),
    testing::ValuesIn(inputLayouts2D),
    testing::ValuesIn(inputShapes2D),
    testing::ValuesIn(axis2D),
    testing::Values(CommonTestUtils::DEVICE_CPU),
    testing::Values(std::map<std::string, std::string>())
);

INSTANTIATE_TEST_CASE_P(
    SoftMax2D,
    SoftMaxLayerTest,
    params2D,
    SoftMaxLayerTest::getTestCaseName
);

}  // namespace
