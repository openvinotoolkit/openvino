// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "subgraph_tests/split_trivial_permute_concat.hpp"
#include "common_test_utils/test_constants.hpp"
#include "gna/gna_config.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
    std::vector<InferenceEngine::Precision> netPrecisions = { InferenceEngine::Precision::FP32,
    };

    std::map<std::string, std::string> config = {
            {"GNA_COMPACT_MODE", "NO"}
    };

    std::vector<std::vector<size_t>> inputSizes = {
        { 4, 2, 64, 6 },
        { 4, 16, 4, 128},
        { 2, 10, 16, 64},
        { 2, 32, 64, 2},
    };

    std::vector<size_t> split_axes = { 1 }; // only channels split is currently supported by gna for 4d inputs
    std::vector<size_t> concat_axes = { 1 }; // only channels concat is currently supported by gna for 4d inputs

    INSTANTIATE_TEST_SUITE_P(smoke_split_trivial_permute_concat, SplitTrivialPermuteConcatTest,
        ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_GNA),
            ::testing::ValuesIn(inputSizes),
            ::testing::ValuesIn(split_axes), // split axis
            ::testing::ValuesIn(concat_axes), // concat axis
            ::testing::Values(config)),
        SplitTrivialPermuteConcatTest::getTestCaseName);
}  // namespace
