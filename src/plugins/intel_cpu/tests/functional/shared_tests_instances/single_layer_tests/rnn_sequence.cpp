// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "openvino/op/util/attr_types.hpp"
#include "single_op_tests/rnn_sequence.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::RNNSequenceTest;
using ov::test::utils::SequenceTestsMode;
using ov::test::utils::InputLayerType;
using ov::op::RecurrentSequenceDirection;

namespace {
    std::vector<SequenceTestsMode> mode{SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_CONST,
                                        SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_CONST,
                                        SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_PARAM,
                                        SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM,
                                        SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_CONST,
                                        SequenceTestsMode::PURE_SEQ};
    // output values increase rapidly without clip, so use only seq_lengths = 2
    std::vector<size_t> seq_lengths_zero_clip{10};
    std::vector<size_t> seq_lengths_clip_non_zero{20};
    std::vector<size_t> batch{1, 10};
    std::vector<size_t> hidden_size{1, 10};
    std::vector<size_t> input_size{10};
    std::vector<std::vector<std::string>> activations = {{"relu"}, {"sigmoid"}, {"tanh"}};
    std::vector<float> clip{0.f};
    std::vector<float> clip_non_zeros{0.7f};
    std::vector<RecurrentSequenceDirection> direction = {RecurrentSequenceDirection::FORWARD,
                                                         RecurrentSequenceDirection::REVERSE,
                                                         RecurrentSequenceDirection::BIDIRECTIONAL,
    };
    std::vector<ov::element::Type> model_types = {ov::element::f32};

    INSTANTIATE_TEST_SUITE_P(smoke_RNNSequenceCommonZeroClip, RNNSequenceTest,
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
                            RNNSequenceTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_RNNSequenceCommonClip, RNNSequenceTest,
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
                            RNNSequenceTest::getTestCaseName);

}  // namespace
