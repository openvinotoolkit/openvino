// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/rnn_cell.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
    std::vector<bool> should_decompose{false, true};
    std::vector<size_t> batch{1, 5};
    std::vector<size_t> hidden_size{1, 10};
    std::vector<size_t> input_size{1, 30};
    std::vector<std::vector<std::string>> activations = {{"relu"}, {"sigmoid"}, {"tanh"}};
    std::vector<float> clip = {0.f, 0.7f};
    std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                             InferenceEngine::Precision::FP16};

    INSTANTIATE_TEST_SUITE_P(RNNCellCommon, RNNCellTest,
            ::testing::Combine(
            ::testing::ValuesIn(should_decompose),
            ::testing::ValuesIn(batch),
            ::testing::ValuesIn(hidden_size),
            ::testing::ValuesIn(input_size),
            ::testing::ValuesIn(activations),
            ::testing::ValuesIn(clip),
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_GPU)),
            RNNCellTest::getTestCaseName);

}  // namespace
