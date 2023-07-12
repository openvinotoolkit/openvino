// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/split_conv.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::map<std::string, std::string>> configs = {{{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}},
                                                                 {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}}};

std::vector<convParams> params = {std::make_tuple(std::vector<size_t>{1, 128},  // InputShape
                                                  std::vector<size_t>{1, 3},    // KernelShape
                                                  1),                           // Stride
                                  std::make_tuple(std::vector<size_t>{1, 256}, std::vector<size_t>{1, 5}, 1),
                                  std::make_tuple(std::vector<size_t>{1, 336}, std::vector<size_t>{1, 9}, 2),
                                  std::make_tuple(std::vector<size_t>{1, 640}, std::vector<size_t>{1, 8}, 4)};

std::vector<size_t> inputChannels = {1, 4, 8};

std::vector<size_t> outputChannels = {4, 8};

INSTANTIATE_TEST_SUITE_P(smoke_SplitConvTest,
                         SplitConvTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(params),
                                            ::testing::ValuesIn(inputChannels),
                                            ::testing::ValuesIn(outputChannels)),
                         SplitConvTest::getTestCaseName);
}  // namespace
