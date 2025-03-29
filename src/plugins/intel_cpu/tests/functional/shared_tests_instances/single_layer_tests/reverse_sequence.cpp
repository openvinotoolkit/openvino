// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/reverse_sequence.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::ReverseSequenceLayerTest;

namespace {

const std::vector<ov::element::Type> model_types = {
        ov::element::f32,
        ov::element::f16,
        ov::element::u8,
        ov::element::i8,
        ov::element::u16,
        ov::element::i32
};

const std::vector<int64_t> batch_axis_indices = { 0L };

const std::vector<int64_t> seq_axis_indices = { 1L };

const std::vector<std::vector<size_t>> input_shapes = { {3, 10} }; //, 10, 20

const std::vector<std::vector<size_t>> reverse_seq_shapes = { {3} };

const std::vector<ov::test::utils::InputLayerType> secondary_input_types = {
        ov::test::utils::InputLayerType::CONSTANT,
        ov::test::utils::InputLayerType::PARAMETER
};

INSTANTIATE_TEST_SUITE_P(smoke_ReverseSequence, ReverseSequenceLayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(batch_axis_indices),
                            ::testing::ValuesIn(seq_axis_indices),
                            ::testing::ValuesIn(input_shapes),
                            ::testing::ValuesIn(reverse_seq_shapes),
                            ::testing::ValuesIn(secondary_input_types),
                            ::testing::ValuesIn(model_types),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        ReverseSequenceLayerTest::getTestCaseName);

}  // namespace
