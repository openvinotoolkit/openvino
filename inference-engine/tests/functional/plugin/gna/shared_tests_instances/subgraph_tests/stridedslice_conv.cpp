// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "subgraph_tests/stridedslice_conv.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"}
    }
};

std::vector<convParams> params = {
    std::make_tuple(
         std::vector<size_t>{1, 1, 1, 256},   //InputShape
         std::vector<size_t>{1, 3},           //KernelShape
         1),                                  //Stride
    std::make_tuple(std::vector<size_t>{1, 1, 1, 1024}, std::vector<size_t>{1, 5}, 1),
    std::make_tuple(std::vector<size_t>{1, 1, 1, 336}, std::vector<size_t>{1, 9}, 2),
    std::make_tuple(std::vector<size_t>{1, 1, 1, 640}, std::vector<size_t>{1, 8}, 4)
};

std::vector<size_t> outputChannels = {
    4,
    8
};

INSTANTIATE_TEST_SUITE_P(smoke_SliceConvTest, SliceConvTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                            ::testing::ValuesIn(configs),
                            ::testing::ValuesIn(params),
                            ::testing::ValuesIn(outputChannels)),
                        SliceConvTest::getTestCaseName);
}  // namespace
