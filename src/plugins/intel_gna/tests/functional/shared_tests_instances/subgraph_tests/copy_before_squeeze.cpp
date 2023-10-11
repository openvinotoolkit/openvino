// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/copy_before_squeeze.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "gna/gna_config.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16,
                                                         InferenceEngine::Precision::FP32};

std::map<std::string, std::string> config = {{"GNA_COMPACT_MODE", "NO"}};

std::vector<std::vector<size_t>> inputShapes = {{1, 512}, {1, 1024}, {1, 192}, {1, 640}};

INSTANTIATE_TEST_SUITE_P(smoke_copy_before_squeeze,
                         CopyBeforeSqueezeTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(inputShapes),
                                            ::testing::Values(config)),
                         CopyBeforeSqueezeTest::getTestCaseName);
}  // namespace
