// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <ngraph/op/util/attr_types.hpp>
#include "single_layer_tests/rnn_sequence.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
    // output values increase rapidly without clip, so use only seq_lenghts = 2
    std::vector<size_t> seq_lengths_zero_clip{2};
    std::vector<size_t> seq_lengths_clip_non_zero{20};
    std::vector<size_t> batch{1, 10};
    std::vector<size_t> hidden_size{1, 10};
    std::vector<size_t> input_size{10};
    std::vector<std::vector<std::string>> activations = {{"relu"}, {"sigmoid"}, {"tanh"}};
    std::vector<float> clip{0.f};
    std::vector<float> clip_non_zeros{0.7f};
    std::vector<ngraph::op::RecurrentSequenceDirection> direction = {ngraph::op::RecurrentSequenceDirection::FORWARD,
                                                           ngraph::op::RecurrentSequenceDirection::REVERSE,
                                                           ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL
    };
    std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                             InferenceEngine::Precision::FP16};

    INSTANTIATE_TEST_CASE_P(smoke_RNNSequenceCommonZeroClip, RNNSequenceTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(seq_lengths_zero_clip),
                                    ::testing::ValuesIn(batch),
                                    ::testing::ValuesIn(hidden_size),
                                    ::testing::ValuesIn(input_size),
                                    ::testing::ValuesIn(activations),
                                    ::testing::ValuesIn(clip),
                                    ::testing::ValuesIn(direction),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            RNNSequenceTest::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_RNNSequenceCommonClip, RNNSequenceTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(seq_lengths_clip_non_zero),
                                    ::testing::ValuesIn(batch),
                                    ::testing::ValuesIn(hidden_size),
                                    ::testing::ValuesIn(input_size),
                                    ::testing::ValuesIn(activations),
                                    ::testing::ValuesIn(clip_non_zeros),
                                    ::testing::ValuesIn(direction),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            RNNSequenceTest::getTestCaseName);

}  // namespace
