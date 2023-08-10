// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/trivial_concat.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
std::vector<std::vector<size_t>> inShapes = {
    {1, 1, 33, 16},
    {1, 1, 65, 16},
    {10, 16},
    {10, 64},
    {15, 15},
};

std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                         InferenceEngine::Precision::FP16};

std::map<std::string, std::string> additional_config = {
    {"GNA_COMPACT_MODE", "NO"},
    {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
    {"GNA_SCALE_FACTOR_0", "2000.0"},
};

INSTANTIATE_TEST_SUITE_P(smoke_trivial_concat_Basic,
                         TrivialConcatLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inShapes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::Values(additional_config)),
                         TrivialConcatLayerTest::getTestCaseName);
}  // namespace
