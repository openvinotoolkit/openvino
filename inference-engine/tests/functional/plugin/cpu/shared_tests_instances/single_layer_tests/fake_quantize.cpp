// Copyright (C) 2020 Intel Corporation
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

const std::vector<std::vector<size_t>> inputShapes = {{1, 1, 1, 1}, {3, 10, 5, 6}};
const std::vector<std::vector<size_t>> constShapes = {{1}};
const std::vector<size_t> levels = {16, 255, 256};

const std::pair<std::string, std::map<std::string, std::string>> config = {};
const std::vector<float> fqArgs = {};
const std::vector<float> inputParams = {};


const auto fqParams = ::testing::Combine(
        ::testing::ValuesIn(levels),
        ::testing::ValuesIn(constShapes),
        ::testing::Values(fqArgs),
        ::testing::Values(inputParams)
);

INSTANTIATE_TEST_CASE_P(smoke_FakeQuantize, FakeQuantizeLayerTest,
                        ::testing::Combine(
                                fqParams,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(inputShapes),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(config)),
                        FakeQuantizeLayerTest::getTestCaseName);

}  // namespace
