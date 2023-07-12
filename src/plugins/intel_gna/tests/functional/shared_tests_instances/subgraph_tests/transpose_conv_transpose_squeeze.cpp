// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/transpose_conv_transpose_squeeze.hpp"

#include <gna/gna_config.hpp>
#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

const std::vector<std::map<std::string, std::string>> configs = {{{"GNA_DEVICE_MODE", "GNA_SW_FP32"}},
                                                                 {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}}};

const std::vector<std::vector<size_t>> inputShapes = {{1, 8192}};

const std::vector<std::vector<size_t>> kernels = {{1, 3}, {1, 4}, {1, 8}, {1, 9}};
const std::vector<std::vector<size_t>> strides = {{1, 1}};
const std::vector<size_t> inputChannels = {64};
const std::vector<size_t> outputChannels{4, 8, 16};

const auto convParams = ::testing::Combine(::testing::ValuesIn(kernels),
                                           ::testing::ValuesIn(strides),
                                           ::testing::ValuesIn(inputChannels),
                                           ::testing::ValuesIn(outputChannels));

INSTANTIATE_TEST_SUITE_P(smoke_TransposeConvTest,
                         TransposeConvTest,
                         ::testing::Combine(convParams,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(inputShapes),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         TransposeConvTest::getTestCaseName);

}  // namespace
