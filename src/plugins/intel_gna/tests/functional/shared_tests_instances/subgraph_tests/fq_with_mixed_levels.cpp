// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/fq_with_mixed_levels.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

namespace SubgraphTestsDefinitions {
namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::map<std::string, std::string>> configs = {{{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}}};

INSTANTIATE_TEST_SUITE_P(smoke_FqWithMixedLevelsTest,
                         FqWithMixedLevelsTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         FqWithMixedLevelsTest::getTestCaseName);
}  // namespace
}  // namespace SubgraphTestsDefinitions
