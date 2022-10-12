// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/lstm_cell_basic.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
    std::vector<bool> should_decompose{false, true};
    std::vector<size_t> batch{1};
    std::vector<size_t> hidden_size{1, 5, 16};
    std::vector<size_t> input_size{1, 10, 16};
    std::vector<std::vector<std::string>> activations = {{"sigmoid", "tanh", "tanh"},
                                                         {"sigmoid", "sigmoid", "sigmoid"},
                                                         {"tanh", "tanh", "tanh"},
                                                         {"tanh", "sigmoid", "relu"}};
    float clip = 0.f;
    std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                             InferenceEngine::Precision::FP16};

    std::vector<std::map<std::string, std::string>> configs = {
        {
            {"GNA_DEVICE_MODE", "GNA_SW_FP32"}
        },
        {
            {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
            {"GNA_PRECISION", "I16"},
            {"GNA_SCALE_FACTOR_0", "1024"},
            {"GNA_SCALE_FACTOR_1", "1024"},
            {"GNA_SCALE_FACTOR_2", "1024"}
        },
        {
            {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
            {"GNA_PRECISION", "I8"},
            {"GNA_SCALE_FACTOR_0", "1024"},
            {"GNA_SCALE_FACTOR_1", "1024"},
            {"GNA_SCALE_FACTOR_2", "1024"}
        }
    };

    INSTANTIATE_TEST_SUITE_P(smoke_LSTMCellBasicCommon, LSTMCellBasicTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(should_decompose),
                                    ::testing::ValuesIn(batch),
                                    ::testing::ValuesIn(hidden_size),
                                    ::testing::ValuesIn(input_size),
                                    ::testing::ValuesIn(activations),
                                    ::testing::Values(clip),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                    ::testing::ValuesIn(configs)),
                            LSTMCellBasicTest::getTestCaseName);

} // namespace
