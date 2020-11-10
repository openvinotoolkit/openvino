// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <ngraph/op/util/attr_types.hpp>
#include "single_layer_tests/tensor_iterator.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
    // output values increase rapidly without clip, so use only seq_lenghts = 2
    std::vector<bool> should_decompose = {true, false};
    std::vector<size_t> seq_lengths_zero_clip{2};
    std::vector<size_t> seq_lengths_clip_non_zero{20};
    std::vector<size_t> batch{1, 10};
    std::vector<size_t> hidden_size{1, 10};
    std::vector<size_t> input_size{10};
    std::vector<ngraph::helpers::TensorIteratorBody> body_type
        = {ngraph::helpers::TensorIteratorBody::LSTM, ngraph::helpers::TensorIteratorBody::RNN,
           ngraph::helpers::TensorIteratorBody::GRU};
    std::vector<float> clip{0.f};
    std::vector<float> clip_non_zeros{0.7f};
    std::vector<ngraph::op::RecurrentSequenceDirection> direction = {ngraph::op::RecurrentSequenceDirection::FORWARD,
                                                           ngraph::op::RecurrentSequenceDirection::REVERSE};
    std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                             InferenceEngine::Precision::FP16};

    INSTANTIATE_TEST_CASE_P(smoke_TensorIteratorCommon, TensorIteratorTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(should_decompose),
                                    ::testing::ValuesIn(seq_lengths_zero_clip),
                                    ::testing::ValuesIn(batch),
                                    ::testing::ValuesIn(hidden_size),
                                    ::testing::ValuesIn(input_size),
                                    ::testing::ValuesIn(clip),
                                    ::testing::ValuesIn(body_type),
                                    ::testing::ValuesIn(direction),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            TensorIteratorTest::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_TensorIteratorCommonClip, TensorIteratorTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(should_decompose),
                                    ::testing::ValuesIn(seq_lengths_clip_non_zero),
                                    ::testing::ValuesIn(batch),
                                    ::testing::ValuesIn(hidden_size),
                                    ::testing::ValuesIn(input_size),
                                    ::testing::ValuesIn(clip_non_zeros),
                                    ::testing::ValuesIn(body_type),
                                    ::testing::ValuesIn(direction),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            TensorIteratorTest::getTestCaseName);

}  // namespace
