// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "subgraph_tests/eltwise_conv_eltwise.hpp"

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
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"}
    }
};

std::vector<convParams> params = {
    std::make_tuple(
         std::vector<size_t>{1, 64},    //InputShape
         std::vector<size_t>{1, 3},     //KernelShape
         1),                            //Stride
    std::make_tuple(std::vector<size_t>{1, 128}, std::vector<size_t>{1, 5}, 1),
    std::make_tuple(std::vector<size_t>{1, 168}, std::vector<size_t>{1, 9}, 2),
    std::make_tuple(std::vector<size_t>{1, 320}, std::vector<size_t>{1, 8}, 4)
};

std::vector<size_t> inputChannels = {
    1,
    4,
    8
};

std::vector<size_t> outputChannels = {
    4,
    8
};

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseAfterConvTest, EltwiseAfterConvTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                            ::testing::ValuesIn(configs),
                            ::testing::ValuesIn(params),
                            ::testing::ValuesIn(inputChannels),
                            ::testing::ValuesIn(outputChannels)),
                        EltwiseAfterConvTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseBeforeConvTest, EltwiseBeforeConvTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                            ::testing::ValuesIn(configs),
                            ::testing::ValuesIn(params),
                            ::testing::ValuesIn(inputChannels),
                            ::testing::ValuesIn(outputChannels)),
                        EltwiseBeforeConvTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseWithTwoConvsAsInputsTest, EltwiseWithTwoConvsAsInputsTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                            ::testing::ValuesIn(configs),
                            ::testing::ValuesIn(params),
                            ::testing::ValuesIn(inputChannels),
                            ::testing::ValuesIn(outputChannels)),
                        EltwiseWithTwoConvsAsInputsTest::getTestCaseName);
}  // namespace
