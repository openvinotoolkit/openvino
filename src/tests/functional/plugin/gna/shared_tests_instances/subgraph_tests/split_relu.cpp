// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "subgraph_tests/split_relu.hpp"
#include "common_test_utils/test_constants.hpp"
#include "gna/gna_config.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
    std::vector<std::vector<std::vector<size_t>>> inputs{
            {{1, 64}},
            {{1, 128}},
            {{1, 96}},
            {{1, 16}}
    };

    std::vector<std::vector<size_t>> connect_index{
            {0, 1, 2, 3},
            {0, 1, 2},
            {0, 1, 3},
            {0, 2, 3},
            {1, 2, 3},
            {0, 1},
            {0, 2},
            {0, 3},
            {1, 2},
            {1, 3},
            {2, 3},
            {0},
            {1},
            {2},
            {3},
    };

    std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                             InferenceEngine::Precision::FP16,
    };

    std::map<std::string, std::string> additional_config = {
            {"GNA_SCALE_FACTOR_1", "1"},
            {"GNA_SCALE_FACTOR_2", "1"},
            {"GNA_SCALE_FACTOR_3", "1"},
            {"GNA_SCALE_FACTOR_4", "1"},
            {std::string(GNA_CONFIG_KEY(COMPACT_MODE)), "NO"}
    };

    INSTANTIATE_TEST_SUITE_P(smoke_split_connected, SplitRelu,
                            ::testing::Combine(
                                    ::testing::ValuesIn(inputs),
                                    ::testing::ValuesIn(connect_index),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                    ::testing::Values(additional_config)),
                            SplitRelu::getTestCaseName);
}  // namespace
