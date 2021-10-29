// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <subgraph_tests/memory_eltwise_reshape_concat.hpp>
#include "common_test_utils/test_constants.hpp"

namespace SubgraphTestsDefinitions {
namespace {
std::vector<size_t> input_multiples = {
    1,
    7,
    5,
    8
};

std::vector<size_t> concat_sizes = {
    32,
    64
};

std::map<std::string, std::string> additional_config = {
    {"GNA_COMPACT_MODE", "NO"},
    {"GNA_DEVICE_MODE", "GNA_SW_FP32"},
    {"GNA_SCALE_FACTOR_0", "1638.4"},
};
} // namespace

INSTANTIATE_TEST_SUITE_P(smoke_MemoryEltwiseReshapeConcatTest, MemoryEltwiseReshapeConcatTest,
    ::testing::Combine(
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::ValuesIn(input_multiples),
        ::testing::ValuesIn(concat_sizes),
        ::testing::Values(additional_config)),
    MemoryEltwiseReshapeConcatTest::getTestCaseName);
} // namespace SubgraphTestsDefinitions
