// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/lstm_cell.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
std::vector<bool> should_decompose{false, true};
std::vector<size_t> batch{5};
std::vector<size_t> hidden_size{1, 10};
std::vector<size_t> hidden_size_smoke{1};
std::vector<size_t> input_size{1, 30};
std::vector<std::vector<std::string>> activations_smoke = {{"relu", "sigmoid", "tanh"}};
std::vector<std::vector<std::string>> activations = {{"relu", "sigmoid", "tanh"}, {"sigmoid", "tanh", "tanh"},
                                                     {"tanh", "relu", "sigmoid"}, {"sigmoid", "sigmoid", "sigmoid"},
                                                     {"tanh", "tanh", "tanh"}, {"relu", "relu", "relu"}};
std::vector<float> clip{0.f, 0.7f};
std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                        InferenceEngine::Precision::FP16};

INSTANTIATE_TEST_SUITE_P(LSTMCellCommon, LSTMCellTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(should_decompose),
                                ::testing::ValuesIn(batch),
                                ::testing::ValuesIn(hidden_size),
                                ::testing::ValuesIn(input_size),
                                ::testing::ValuesIn(activations),
                                ::testing::ValuesIn(clip),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        LSTMCellTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_LSTMCellCommon, LSTMCellTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(should_decompose),
                                ::testing::ValuesIn(batch),
                                ::testing::ValuesIn(hidden_size_smoke),
                                ::testing::ValuesIn(input_size),
                                ::testing::ValuesIn(activations_smoke),
                                ::testing::ValuesIn(clip),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        LSTMCellTest::getTestCaseName);
}  // namespace
