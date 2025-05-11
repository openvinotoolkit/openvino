// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/lstm_sequence.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::LSTMSequenceTest;
using ov::test::utils::SequenceTestsMode;
using ov::test::utils::InputLayerType;

std::vector<SequenceTestsMode> mode{
    SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_CONST,
    SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_CONST,
    SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_PARAM,
    SequenceTestsMode::PURE_SEQ,
    SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_CONST,
    SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM};

// output values increase rapidly without clip, so use only seq_lengths = 2
std::vector<size_t> seq_lengths_zero_clip{2};
std::vector<size_t> seq_lengths_clip_non_zero{20};
std::vector<size_t> batch{10};
std::vector<size_t> hidden_size{1, 10};
std::vector<size_t> input_size{10};
std::vector<std::vector<std::string>> activations = {{"relu", "sigmoid", "tanh"}, {"sigmoid", "tanh", "tanh"},
                                                        {"tanh", "relu", "sigmoid"}, {"sigmoid", "sigmoid", "sigmoid"},
                                                        {"tanh", "tanh", "tanh"}, {"relu", "relu", "relu"}};
std::vector<float> clip{0.f};
std::vector<float> clip_non_zeros{0.7f};
std::vector<ov::op::RecurrentSequenceDirection> direction = {ov::op::RecurrentSequenceDirection::FORWARD,
                                                        ov::op::RecurrentSequenceDirection::REVERSE,
                                                        ov::op::RecurrentSequenceDirection::BIDIRECTIONAL
};
std::vector<ov::element::Type> model_types = {
    ov::element::f32,
    ov::element::f16};

INSTANTIATE_TEST_SUITE_P(smoke_LSTMSequenceCommonZeroClip, LSTMSequenceTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(mode),
                                ::testing::ValuesIn(seq_lengths_zero_clip),
                                ::testing::ValuesIn(batch),
                                ::testing::ValuesIn(hidden_size),
                                ::testing::ValuesIn(input_size),
                                ::testing::ValuesIn(activations),
                                ::testing::ValuesIn(clip),
                                ::testing::ValuesIn(direction),
                                ::testing::Values(InputLayerType::CONSTANT),
                                ::testing::ValuesIn(model_types),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        LSTMSequenceTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_LSTMSequenceCommonZeroClipNonconstantWRB, LSTMSequenceTest,
                        ::testing::Combine(
                                ::testing::Values(SequenceTestsMode::PURE_SEQ),
                                ::testing::ValuesIn(seq_lengths_zero_clip),
                                ::testing::ValuesIn(batch),
                                ::testing::ValuesIn(hidden_size),
                                ::testing::ValuesIn(input_size),
                                ::testing::ValuesIn(activations),
                                ::testing::ValuesIn(clip),
                                ::testing::ValuesIn(direction),
                                ::testing::Values(InputLayerType::PARAMETER),
                                ::testing::ValuesIn(model_types),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        LSTMSequenceTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_LSTMSequenceCommonClip, LSTMSequenceTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(mode),
                                ::testing::ValuesIn(seq_lengths_clip_non_zero),
                                ::testing::ValuesIn(batch),
                                ::testing::ValuesIn(hidden_size),
                                ::testing::ValuesIn(input_size),
                                ::testing::ValuesIn(activations),
                                ::testing::ValuesIn(clip_non_zeros),
                                ::testing::ValuesIn(direction),
                                ::testing::Values(InputLayerType::CONSTANT),
                                ::testing::ValuesIn(model_types),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        LSTMSequenceTest::getTestCaseName);

}  // namespace
