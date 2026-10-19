// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>
#include <vector>
#include "single_op_tests/gru_sequence.hpp"
#include "common_test_utils/test_constants.hpp"
#include "common_test_utils/test_enums.hpp"

using ov::test::GRUSequenceTest;
using ov::test::utils::InputLayerType;
using ov::test::utils::SequenceTestsMode;

namespace {
    class GRUSequenceNoClipCPUTest : public GRUSequenceTest {};

    TEST_P(GRUSequenceNoClipCPUTest, InferenceKeepsSequencePrimitive) {
        run();
        ov::test::CheckNumberOfNodesWithType(compiledModel, "RNNSeq", 1);
    }

    std::vector<SequenceTestsMode> mode{SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_CONST,
                                        SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_CONST,
                                        SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_PARAM,
                                        SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM,
                                        SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_CONST,
                                        SequenceTestsMode::PURE_SEQ};
    // output values increase rapidly without clip, so use only seq_lengths = 2

    const std::vector<std::vector<ov::Shape>> input_shapes_zero_clip_static = {
    // {batch, seq_lengths, input_size}, {batch, num_directions, hidden_size}, {batch},
        {{ 10, 2, 1}, { 10, 1, 1 }, { 10 }},
        {{ 10, 2, 1}, { 10, 1, 10 }, { 10 }},
    };
    const std::vector<std::vector<ov::Shape>> input_shapes_bidirect_zero_clip_static = {
        {{ 10, 2, 1}, { 10, 2, 1 }, { 10 }},
        {{ 10, 2, 1}, { 10, 2, 10 }, { 10 }},
    };
    const std::vector<std::vector<ov::Shape>> input_shapes_non_zero_clip_static = {
        {{ 10, 20, 1}, { 10, 1, 1 }, { 10 }},
        {{ 10, 20, 1}, { 10, 1, 10 }, { 10 }},
    };
    const std::vector<std::vector<ov::Shape>> input_shapes_bidirect_non_zero_clip_static = {
        {{ 10, 20, 1}, { 10, 2, 1 }, { 10 }},
        {{ 10, 20, 1}, { 10, 2, 10 }, { 10 }},
    };
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
    std::vector<ov::op::RecurrentSequenceDirection> direction = {ov::op::RecurrentSequenceDirection::FORWARD,
                                                                 ov::op::RecurrentSequenceDirection::REVERSE};
    std::vector<ov::op::RecurrentSequenceDirection> direction_bi = {ov::op::RecurrentSequenceDirection::BIDIRECTIONAL};

    std::vector<ov::element::Type> netPrecisions = {ov::element::f32,
                                                    ov::element::f16};

    INSTANTIATE_TEST_SUITE_P(smoke_GRUSequenceCommonZeroClip, GRUSequenceTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(mode),
                                    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_zero_clip_static)),
                                    // ::testing::ValuesIn(input_size), // hardcoded to 10 due to Combine supports up to 10 args
                                    ::testing::ValuesIn(activations),
                                    ::testing::ValuesIn(clip),
                                    ::testing::ValuesIn(linear_before_reset),
                                    ::testing::ValuesIn(direction),
                                    ::testing::Values(InputLayerType::CONSTANT),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(ov::test::utils::DEVICE_CPU)),
                            GRUSequenceTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_GRUSequenceCommonZeroClipBidirect, GRUSequenceTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(mode),
                                    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_bidirect_zero_clip_static)),
                                    // ::testing::ValuesIn(input_size), // hardcoded to 10 due to Combine supports up to 10 args
                                    ::testing::ValuesIn(activations),
                                    ::testing::ValuesIn(clip),
                                    ::testing::ValuesIn(linear_before_reset),
                                    ::testing::ValuesIn(direction_bi),
                                    ::testing::Values(InputLayerType::CONSTANT),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(ov::test::utils::DEVICE_CPU)),
                            GRUSequenceTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_GRUSequenceCommonClip, GRUSequenceTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(mode),
                                    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_non_zero_clip_static)),
                                    // ::testing::ValuesIn(input_size),  // hardcoded to 10 due to Combine supports up to 10 args
                                    ::testing::ValuesIn(activations),
                                    ::testing::ValuesIn(clip_non_zeros),
                                    ::testing::ValuesIn(linear_before_reset),
                                    ::testing::ValuesIn(direction),
                                    ::testing::Values(InputLayerType::CONSTANT),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(ov::test::utils::DEVICE_CPU)),
                            GRUSequenceTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_GRUSequenceCommonClipBidirect, GRUSequenceTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(mode),
                                    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_bidirect_non_zero_clip_static)),
                                    // ::testing::ValuesIn(input_size),  // hardcoded to 10 due to Combine supports up to 10 args
                                    ::testing::ValuesIn(activations),
                                    ::testing::ValuesIn(clip_non_zeros),
                                    ::testing::ValuesIn(linear_before_reset),
                                    ::testing::ValuesIn(direction_bi),
                                    ::testing::Values(InputLayerType::CONSTANT),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(ov::test::utils::DEVICE_CPU)),
                            GRUSequenceTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_GRUSequenceNoClip, GRUSequenceNoClipCPUTest,
                            ::testing::Combine(
                                    ::testing::Values(SequenceTestsMode::PURE_SEQ),
                                    ::testing::Values(ov::test::static_shapes_to_test_representation(
                                            input_shapes_zero_clip_static.front())),
                                    ::testing::Values(std::vector<std::string>{"sigmoid", "tanh"}),
                                    ::testing::Values(std::numeric_limits<float>::infinity(),
                                                      -1.f,
                                                      -std::numeric_limits<float>::infinity(),
                                                      std::numeric_limits<float>::quiet_NaN()),
                                    ::testing::Values(true),
                                    ::testing::Values(ov::op::RecurrentSequenceDirection::FORWARD),
                                    ::testing::Values(InputLayerType::CONSTANT),
                                    ::testing::Values(ov::element::f32),
                                    ::testing::Values(ov::test::utils::DEVICE_CPU)),
                            GRUSequenceTest::getTestCaseName);

}  // namespace
