// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "subgraph_tests/matmul_squeeze_add.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "81.9175"}
    }
};

std::vector<std::vector<size_t>> input_shapes = {
    {1, 8},
    {1, 42},
    {1, 100},
    {1, 128},
    {1, 512}
};

std::vector<size_t> output_sizes = {
    1000,
    512,
    128,
    42,
    16,
    8
};

INSTANTIATE_TEST_SUITE_P(smoke_MatmulSqueezeAdd, MatmulSqueezeAddTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                            ::testing::ValuesIn(configs),
                            ::testing::ValuesIn(input_shapes),
                            ::testing::ValuesIn(output_sizes)),
                        MatmulSqueezeAddTest::getTestCaseName);
}  // namespace
