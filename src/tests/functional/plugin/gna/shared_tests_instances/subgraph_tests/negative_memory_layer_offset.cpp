// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "subgraph_tests/negative_memory_layer_offset.hpp"
#include "common_test_utils/test_constants.hpp"
#include "gna/gna_config.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
    std::vector<InferenceEngine::Precision> netPrecisions = { InferenceEngine::Precision::FP32,
    };

    std::map<std::string, std::string> config = {
            {"GNA_COMPACT_MODE", "NO"}
    };

    std::vector<size_t> inputSizes = {
        384,
        128,
        64,
        32
    };

    std::vector<size_t> hiddenSizes = {
        384,
        128,
        64,
        32,
        100
    };

    INSTANTIATE_TEST_SUITE_P(smoke_negative_memory_layer_offset, NegativeMemoryOffsetTest,
        ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_GNA),
            ::testing::ValuesIn(inputSizes),
            ::testing::ValuesIn(hiddenSizes),
            ::testing::Values(config)),
        NegativeMemoryOffsetTest::getTestCaseName);
}  // namespace
