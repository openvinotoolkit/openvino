// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "subgraph_tests/input_conv.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
    {
        {"GNA_DEVICE_MODE", "GNA_SW_FP32"}
    },
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "163.835"}
    }
};

std::vector<convParams> params = {
    std::make_tuple(
         std::vector<size_t>{1, 1, 1, 16},    //InputShape
         std::vector<size_t>{1, 8},           //KernelShape
         1),                                  //Stride
    std::make_tuple(std::vector<size_t>{1, 1, 1, 16}, std::vector<size_t>{1, 9}, 1),
    std::make_tuple(std::vector<size_t>{1, 1, 1, 168}, std::vector<size_t>{1, 9}, 1),
    std::make_tuple(std::vector<size_t>{1, 1, 1, 168}, std::vector<size_t>{1, 8}, 1),
    std::make_tuple(std::vector<size_t>{1, 1, 1, 640}, std::vector<size_t>{1, 512}, 128)
};

std::vector<size_t> outputChannels = {
    4,
    8
};

std::vector<bool> addReshape = {
    true,
    false
};

INSTANTIATE_TEST_SUITE_P(smoke_InputConv, InputConvTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                            ::testing::ValuesIn(configs),
                            ::testing::ValuesIn(params),
                            ::testing::ValuesIn(outputChannels),
                            ::testing::ValuesIn(addReshape)),
                        InputConvTest::getTestCaseName);
}  // namespace
