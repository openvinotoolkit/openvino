// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/lstm_cell.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
    std::vector<bool> should_decompose{false, true};
    std::vector<size_t> batch{1};
    std::vector<size_t> hidden_size{1, 5};
    std::vector<size_t> input_size{1, 10};
    std::vector<std::vector<std::string>> activations = {{"sigmoid", "tanh", "tanh"},
                                                         {"sigmoid", "sigmoid", "sigmoid"},
                                                         {"tanh", "tanh", "tanh"}};
    float clip = 0.f;
    std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                             InferenceEngine::Precision::FP16};

    INSTANTIATE_TEST_SUITE_P(smoke_LSTMCellCommon, LSTMCellTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(should_decompose),
                                    ::testing::ValuesIn(batch),
                                    ::testing::ValuesIn(hidden_size),
                                    ::testing::ValuesIn(input_size),
                                    ::testing::ValuesIn(activations),
                                    ::testing::Values(clip),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                            LSTMCellTest::getTestCaseName);

} // namespace
