// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/fake_quantize.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<std::vector<size_t>> inputShapes = {{1, 1, 1, 1}, {3, 10, 5, 6}, {1, 2, 3, 4, 2, 3, 2, 2}};
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
        ::testing::Values(ngraph::op::AutoBroadcastType::NUMPY)
);

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantize, FakeQuantizeLayerTest,
                        ::testing::Combine(
                                fqParams,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputShapes),
                                ::testing::Values(ov::test::utils::DEVICE_GPU),
                                ::testing::Values(config)),
                        FakeQuantizeLayerTest::getTestCaseName);

}  // namespace
