// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "subgraph_tests/input_conv.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "910.19"},
        {"GNA_COMPACT_MODE", "NO"}
    }
};

const std::vector<std::map<std::string, bool>> additional_ops = {
    { },
    {
        { "reshape", true}
    }
};

std::vector<std::vector<size_t>> input_shapes = {
    {1, 1, 1, 16},
    {1, 1, 1, 168}
};

std::vector<size_t> output_channels = {
    2,
    4,
    7,
    8
};

INSTANTIATE_TEST_CASE_P(InputConv, InputConvTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                            ::testing::ValuesIn(configs),
                            ::testing::ValuesIn(input_shapes)),
                        InputConvTest::getTestCaseName);
}  // namespace
