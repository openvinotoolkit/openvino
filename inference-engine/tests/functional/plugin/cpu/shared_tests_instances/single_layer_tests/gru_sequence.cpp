// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <ngraph/op/util/attr_types.hpp>
#include "single_layer_tests/gru_sequence.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
    std::vector<ngraph::helpers::SequenceTestsMode> mode{ngraph::helpers::SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_CONST,
                                                         ngraph::helpers::SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_CONST,
                                                         ngraph::helpers::SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_PARAM,
                                                         ngraph::helpers::SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM,
                                                         ngraph::helpers::SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_CONST,
                                                         ngraph::helpers::SequenceTestsMode::PURE_SEQ};
    // output values increase rapidly without clip, so use only seq_lengths = 2
    std::vector<size_t> seq_lengths_zero_clip{2};
    std::vector<size_t> seq_lengths_clip_non_zero{20};
    std::vector<size_t> batch{10};
    std::vector<size_t> hidden_size{1, 10};
    // std::vector<size_t> input_size{10};
    std::vector<std::vector<std::string>> activations = {{"relu", "tanh"}, {"tanh", "sigmoid"}, {"sigmoid", "tanh"},
                                                         {"tanh", "relu"}};
    std::vector<bool> linear_before_reset = {true, false};
    std::vector<float> clip{0.f};
    std::vector<float> clip_non_zeros{0.7f};
    std::vector<ngraph::op::RecurrentSequenceDirection> direction = {ngraph::op::RecurrentSequenceDirection::FORWARD,
                                                           ngraph::op::RecurrentSequenceDirection::REVERSE,
                                                           ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL
    };
    std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                             InferenceEngine::Precision::FP16};

    INSTANTIATE_TEST_CASE_P(smoke_GRUSequenceCommonZeroClip, GRUSequenceTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(mode),
                                    ::testing::ValuesIn(seq_lengths_zero_clip),
                                    ::testing::ValuesIn(batch),
                                    ::testing::ValuesIn(hidden_size),
                                    // ::testing::ValuesIn(input_size), // hardcoded to 10 due to Combine supports up to 10 args
                                    ::testing::ValuesIn(activations),
                                    ::testing::ValuesIn(clip),
                                    ::testing::ValuesIn(linear_before_reset),
                                    ::testing::ValuesIn(direction),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            GRUSequenceTest::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_GRUSequenceCommonClip, GRUSequenceTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(mode),
                                    ::testing::ValuesIn(seq_lengths_clip_non_zero),
                                    ::testing::ValuesIn(batch),
                                    ::testing::ValuesIn(hidden_size),
                                    // ::testing::ValuesIn(input_size),  // hardcoded to 10 due to Combine supports up to 10 args
                                    ::testing::ValuesIn(activations),
                                    ::testing::ValuesIn(clip_non_zeros),
                                    ::testing::ValuesIn(linear_before_reset),
                                    ::testing::ValuesIn(direction),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            GRUSequenceTest::getTestCaseName);

}  // namespace
