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
        {"GNA_DEVICE_MODE", "GNA_SW_FP32"}//,
        //{"GNA_SCALE_FACTOR_0", "163.835"}
    }
};

const std::vector<bool> with_bias = {
    true,
    false
};

std::vector<std::vector<size_t>> input_shapes = {
    {1, 1, 1, 16},
    {1, 1, 1, 168}
};

std::vector<size_t> output_channels = {
    4,
    8
};

INSTANTIATE_TEST_CASE_P(InputConv, InputConvTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                            ::testing::ValuesIn(configs),
                            ::testing::ValuesIn(input_shapes),
                            ::testing::ValuesIn(output_channels),
                            ::testing::ValuesIn(with_bias)),
                        InputConvTest::getTestCaseName);
}  // namespace
