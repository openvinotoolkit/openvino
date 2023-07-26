// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/transpose_add.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
namespace {
std::vector<std::vector<size_t>> input_shapes{{1, 4, 32}, {1, 8, 8}, {1, 7, 8}, {1, 40, 3}, {32, 8}, {8, 2}};

std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

std::map<std::string, std::string> additional_config = {
    {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
};
}  // namespace

namespace SubgraphTestsDefinitions {
INSTANTIATE_TEST_SUITE_P(smoke_basic,
                         TransposeAdd,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(input_shapes),
                                            ::testing::Values(additional_config)),
                         TransposeAdd::getTestCaseName);
}  // namespace SubgraphTestsDefinitions
