// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/paged_attention_relative_bias.hpp"

namespace ov {
namespace test {
namespace {

INSTANTIATE_TEST_SUITE_P(smoke_PagedAttentionRelativeBias,
                         PagedAttentionRelativeBiasTest,
                         ::testing::Combine(::testing::Values(ElementType::f32, ElementType::bf16, ElementType::f16),
                                            ::testing::Values(32),            // head_size
                                            ::testing::Values(1, 4),          // head_num
                                            ::testing::Values(1, 7, 64),      // distances
                                            ::testing::Values(0, 9),          // sliding_window_size
                                            ::testing::Values(1, 3),          // batch_size
                                            ::testing::Values(10, 100, 300),  // seq_len
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         PagedAttentionRelativeBiasTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
