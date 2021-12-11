// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "subgraph_tests/input_split_concat.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"}
    },
    {
        {"GNA_DEVICE_MODE", "GNA_SW_FP32"}
    }
};

const std::vector<std::vector<size_t>> inputShapes = {
    {1, 128},
    {1, 512},
    {1, 320}
};

INSTANTIATE_TEST_SUITE_P(smoke_InputSplitConcat, InputSplitConcatTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                            ::testing::ValuesIn(configs),
                            ::testing::ValuesIn(inputShapes)),
                        InputSplitConcatTest::getTestCaseName);
}  // namespace
