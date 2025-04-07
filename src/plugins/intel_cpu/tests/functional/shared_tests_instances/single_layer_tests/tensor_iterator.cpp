// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_op_tests/tensor_iterator.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::TensorIteratorTest;

namespace {
std::vector<bool> should_decompose = {true, false};
// output values increase rapidly without clip, so use only seq_lengths = 2
std::vector<size_t> seq_lengths_zero_clip{2};
std::vector<size_t> seq_lengths_clip_non_zero{20};
std::vector<size_t> batch{1, 10};
std::vector<size_t> hidden_size{1, 10};
// std::vector<size_t> input_size{10};
std::vector<size_t> sequence_axis{0, 1};
std::vector<ov::test::utils::TensorIteratorBody> body_type
= {ov::test::utils::TensorIteratorBody::LSTM, ov::test::utils::TensorIteratorBody::RNN,
        ov::test::utils::TensorIteratorBody::GRU};
std::vector<float> clip{0.f};
std::vector<float> clip_non_zeros{0.7f};
std::vector<ov::op::RecurrentSequenceDirection> direction = {ov::op::RecurrentSequenceDirection::FORWARD,
                                                                 ov::op::RecurrentSequenceDirection::REVERSE};
std::vector<ov::element::Type> model_types = {ov::element::f32, ov::element::f16};

INSTANTIATE_TEST_SUITE_P(smoke_TensorIteratorCommon, TensorIteratorTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(should_decompose),
                                ::testing::ValuesIn(seq_lengths_zero_clip),
                                ::testing::ValuesIn(batch),
                                ::testing::ValuesIn(hidden_size),
                                //::testing::ValuesIn(input_size), // hardcoded to 10 due to Combine supports up to 10 args
                                ::testing::ValuesIn(sequence_axis),
                                ::testing::ValuesIn(clip),
                                ::testing::ValuesIn(body_type),
                                ::testing::ValuesIn(direction),
                                ::testing::ValuesIn(model_types),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        TensorIteratorTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TensorIteratorCommonClip, TensorIteratorTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(should_decompose),
                                ::testing::ValuesIn(seq_lengths_clip_non_zero),
                                ::testing::ValuesIn(batch),
                                ::testing::ValuesIn(hidden_size),
                                //::testing::ValuesIn(input_size), // hardcoded to 10 due to Combine supports up to 10 args
                                ::testing::ValuesIn(sequence_axis),
                                ::testing::ValuesIn(clip_non_zeros),
                                ::testing::ValuesIn(body_type),
                                ::testing::ValuesIn(direction),
                                ::testing::ValuesIn(model_types),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        TensorIteratorTest::getTestCaseName);

}  // namespace
