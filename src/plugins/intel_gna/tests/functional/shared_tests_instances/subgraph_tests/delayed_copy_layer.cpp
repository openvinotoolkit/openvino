// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/delayed_copy_layer.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "gna/gna_config.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

std::vector<std::map<std::string, std::string>> additional_config = {{{"GNA_DEVICE_MODE", "GNA_SW_FP32"}},
                                                                     {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}}};

std::vector<size_t> memory_sizes = {128, 256, 32};

INSTANTIATE_TEST_SUITE_P(smoke_delayed_copy_layer,
                         DelayedCopyTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(additional_config),
                                            ::testing::ValuesIn(memory_sizes)),
                         DelayedCopyTestBase::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_delayed_copy_layer,
                         DelayedCopyAfterReshapeWithMultipleConnTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(additional_config),
                                            ::testing::ValuesIn(memory_sizes)),
                         DelayedCopyTestBase::getTestCaseName);
}  // namespace
