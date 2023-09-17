// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_precision.hpp>
#include <string>
#include <subgraph_tests/broadcast_power.hpp>
#include <vector>

using namespace SubgraphTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

const std::vector<std::map<std::string, std::string>> configs = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_COMPACT_MODE", "NO"}, {"GNA_SCALE_FACTOR_0", "2048"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}}};

const std::vector<std::vector<std::vector<size_t>>> input_shapes{
    {{1, 8224}, {1, 257, 32}},
    {{2, 8224}, {1, 257, 64}},
    {{4, 8224}, {1, 257, 128}},
    {{8, 128}, {8, 128}},
    {{16, 128}, {16, 128}},
    {{18, 128}, {18, 128}},
    {{1, 16, 1, 128}, {1, 16, 1, 128}},
    {{1, 8, 15, 128}, {1, 8, 15, 128}},
    {{4, 4, 4, 4}, {4, 4, 4, 4}},
    {{1, 4, 4, 128}, {1, 4, 4, 128}}
    // TODO: needed add split over channels
    //        {{8, 8224}},
};

INSTANTIATE_TEST_SUITE_P(PowerBroadcast,
                         BroadcastPowerTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         BroadcastPowerTest::getTestCaseName);

}  // namespace
