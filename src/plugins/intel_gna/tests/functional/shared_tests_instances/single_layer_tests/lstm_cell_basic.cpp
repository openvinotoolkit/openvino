// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/lstm_cell_basic.hpp"

#include <vector>

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

std::vector<std::map<std::string, std::string>> create_configs() {
    const std::map<std::string, std::string> scale_factors = {{"GNA_SCALE_FACTOR_0", "1024"},
                                                              {"GNA_SCALE_FACTOR_1", "1024"},
                                                              {"GNA_SCALE_FACTOR_2", "1024"}};

    const std::pair<std::string, std::string> precision_i8 = {"GNA_PRECISION", "I8"};
    const std::pair<std::string, std::string> precision_i16 = {"GNA_PRECISION", "I16"};

    const std::map<std::string, std::string> config_sw_fp32 = {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}};

    const std::map<std::string, std::string> config_target_3_0 = {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_EXEC_TARGET", "GNA_TARGET_3_0"},
    };

    auto config_target_3_0_i8 = config_target_3_0;
    config_target_3_0_i8.insert(precision_i8);
    config_target_3_0_i8.insert(scale_factors.begin(), scale_factors.end());

    auto config_target_3_0_i16 = config_target_3_0;
    config_target_3_0_i16.insert(precision_i16);
    config_target_3_0_i16.insert(scale_factors.begin(), scale_factors.end());

    // In case of GNA Plugin LSTMCell is decomposed to a graph including Convolution layers.
    //
    // For GNA targets < 3.5 Convolution weights are always set to int16 precision, no matter what GNA_PRECISION is set
    // to. For GNA targets >= 3.5 weights can have int8 or int16 precision based on GNA_PRECISION setting, but it's
    // recommended to use POT, and not the GNA_PRECISION setting, so this test needs to be disabled for int8 precision
    // until POT can be used inside functional tests.
    //
    // This is related to the issue 70675.
    std::map<std::string, std::string> config_target_3_5 = {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_EXEC_TARGET", "GNA_TARGET_3_5"},
    };

    config_target_3_5.insert(precision_i16);
    config_target_3_5.insert(scale_factors.begin(), scale_factors.end());

    return {config_sw_fp32, config_target_3_0_i8, config_target_3_0_i16, config_target_3_5};
}

INSTANTIATE_TEST_SUITE_P(smoke_LSTMCellBasicCommon,
                         LSTMCellBasicTest,
                         ::testing::Combine(::testing::ValuesIn(should_decompose),
                                            ::testing::ValuesIn(batch),
                                            ::testing::ValuesIn(hidden_size),
                                            ::testing::ValuesIn(input_size),
                                            ::testing::ValuesIn(activations),
                                            ::testing::Values(clip),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(create_configs())),
                         LSTMCellBasicTest::getTestCaseName);

}  // namespace
