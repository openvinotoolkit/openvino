// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "subgraph_tests/split_to_concat_with_3_inputs.hpp"

namespace SubgraphTestsDefinitions {
namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

std::map<std::string, std::string> additional_config = {};
//const std::vector<std::map<std::string, std::string>> additional_config = {
//     {
//         {"GNA_DEVICE_MODE", "GNA_SW_EXACT"}
//     },
//     {
//         {"GNA_DEVICE_MODE", "GNA_SW_FP32"}
//     }
//};

const std::vector<std::vector<size_t>> inputShapes = {
    {1, 384},
    {1, 1536},
    {1, 960}
};
} // namespace
INSTANTIATE_TEST_SUITE_P(smoke_SplitConcatWith3Inputs, SplitConcatWith3InputsTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                            ::testing::Values(additional_config),
                            ::testing::ValuesIn(inputShapes)),
                        SplitConcatWith3InputsTest::getTestCaseName);
}  // namespace SubgraphTestsDefinitions