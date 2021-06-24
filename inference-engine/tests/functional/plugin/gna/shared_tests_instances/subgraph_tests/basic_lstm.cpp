// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "subgraph_tests/basic_lstm.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "1638.4"}
    }
};

const std::vector<std::pair<size_t, size_t>> size_params = {
    {49, 118},
    {300, 38},
};

const std::vector<bool> decompose = { false, true };

INSTANTIATE_TEST_SUITE_P(smoke_BasicLSTM, Basic_LSTM_S,
                        ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                            ::testing::ValuesIn(configs),
                            ::testing::ValuesIn(size_params),
                            ::testing::ValuesIn(decompose)),
                        Basic_LSTM_S::getTestCaseName);
}  // namespace
