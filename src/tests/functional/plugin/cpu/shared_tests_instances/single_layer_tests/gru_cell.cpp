// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/gru_cell.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
    std::vector<bool> should_decompose{true};
    std::vector<size_t> batch{1};
    std::vector<size_t> hidden_size{36};
    std::vector<size_t> input_size{1};
    std::vector<std::vector<std::string>> activations = {{"sigmoid", "tanh"}};
    std::vector<float> clip = {0.};
    std::vector<bool> linear_before_reset = {true};
    std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32};

    INSTANTIATE_TEST_SUITE_P(smoke_GRUCellCommon, GRUCellTest,
            ::testing::Combine(
            ::testing::ValuesIn(should_decompose),
            ::testing::ValuesIn(batch),
            ::testing::ValuesIn(hidden_size),
            ::testing::ValuesIn(input_size),
            ::testing::ValuesIn(activations),
            ::testing::ValuesIn(clip),
            ::testing::ValuesIn(linear_before_reset),
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
            GRUCellTest::getTestCaseName);

}  // namespace
