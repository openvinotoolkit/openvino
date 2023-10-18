// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <ngraph/op/util/attr_types.hpp>
#include "single_op_tests/gru_sequence.hpp"
#include "common_test_utils/test_constants.hpp"
#include "common_test_utils/test_enums.hpp"

namespace {
using ov::test::GRUSequenceTest;
using ov::test::utils::InputLayerType;
using ov::test::utils::SequenceTestsMode;

    std::vector<SequenceTestsMode> mode{SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_CONST,
                                        SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_CONST,
                                        SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_PARAM,
                                        SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_CONST,
                                        SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM,
                                        SequenceTestsMode::PURE_SEQ};
    // output values increase rapidly without clip, so use only seq_lengths = 2
    std::vector<size_t> seq_lengths_zero_clip{2};
    std::vector<size_t> seq_lengths_clip_non_zero{20};
    std::vector<size_t> batch{10};
    std::vector<size_t> hidden_size{1, 10};
    std::vector<std::vector<ov::Shape>> input_size{{{10}}};
    std::vector<std::vector<std::string>> activations = {{"relu", "tanh"}, {"tanh", "sigmoid"}, {"sigmoid", "tanh"},
                                                         {"tanh", "relu"}};
    std::vector<bool> linear_before_reset = {true, false};
    std::vector<float> clip{0.f};
    std::vector<float> clip_non_zeros{0.7f};
    std::vector<ov::op::RecurrentSequenceDirection> direction = {ov::op::RecurrentSequenceDirection::FORWARD,
                                                                     ov::op::RecurrentSequenceDirection::REVERSE,
                                                                     ov::op::RecurrentSequenceDirection::BIDIRECTIONAL
    };
    std::vector<ov::element::Type> netPrecisions = {ov::element::f32,
                                                    ov::element::f16};

    INSTANTIATE_TEST_SUITE_P(GRUSequenceCommonZeroClip, GRUSequenceTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(mode),
                                    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_size)),
                                    // hardcoded to 10 due to Combine supports up to 10 args
                                    ::testing::ValuesIn(activations),
                                    ::testing::ValuesIn(clip),
                                    ::testing::ValuesIn(linear_before_reset),
                                    ::testing::ValuesIn(direction),
                                    ::testing::Values(InputLayerType::CONSTANT),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(ov::test::utils::DEVICE_GPU)),
                            GRUSequenceTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(GRUSequenceCommonZeroClipNonConstantWRB, GRUSequenceTest,
                            ::testing::Combine(
                                    ::testing::Values(SequenceTestsMode::PURE_SEQ),
                                    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_size)),
                                    // hardcoded to 10 due to Combine supports up to 10 args
                                    ::testing::ValuesIn(activations),
                                    ::testing::ValuesIn(clip),
                                    ::testing::ValuesIn(linear_before_reset),
                                    ::testing::ValuesIn(direction),
                                    ::testing::Values(InputLayerType::PARAMETER),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(ov::test::utils::DEVICE_GPU)),
                            GRUSequenceTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(GRUSequenceCommonClip, GRUSequenceTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(mode),
                                    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_size)),
                                    // hardcoded to 10 due to Combine supports up to 10 args
                                    ::testing::ValuesIn(activations),
                                    ::testing::ValuesIn(clip_non_zeros),
                                    ::testing::ValuesIn(linear_before_reset),
                                    ::testing::ValuesIn(direction),
                                    ::testing::Values(InputLayerType::CONSTANT),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(ov::test::utils::DEVICE_GPU)),
                            GRUSequenceTest::getTestCaseName);

}  // namespace
