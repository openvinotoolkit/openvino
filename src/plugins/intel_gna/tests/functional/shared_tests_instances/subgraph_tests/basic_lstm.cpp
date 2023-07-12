// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/basic_lstm.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::map<std::string, std::string>> configs = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_SCALE_FACTOR_0", "1638.4"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}, {"GNA_SCALE_FACTOR_0", "1638.4"}}};

const std::vector<std::pair<size_t, size_t>> size_params = {
    {49, 118},
    {300, 38},
};

size_t small_num_cells = 10;

size_t big_num_cells = 49;

std::pair<float, float> weights_range = {0.f, 0.02f};

const std::vector<bool> decompose = {false, true};

INSTANTIATE_TEST_SUITE_P(smoke_BasicLSTM,
                         Basic_LSTM_S,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(size_params),
                                            ::testing::Values(small_num_cells),
                                            ::testing::ValuesIn(decompose),
                                            ::testing::Values(weights_range)),
                         Basic_LSTM_S::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BasicLSTM_big_cells_num,
                         Basic_LSTM_S,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::Values(size_params[0]),
                                            ::testing::Values(big_num_cells),
                                            ::testing::ValuesIn(decompose),
                                            ::testing::Values(weights_range)),
                         Basic_LSTM_S::getTestCaseName);
}  // namespace
