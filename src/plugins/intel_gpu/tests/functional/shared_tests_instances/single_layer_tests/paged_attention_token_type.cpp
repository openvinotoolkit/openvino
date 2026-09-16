// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/paged_attention_token_type.hpp"

namespace ov {
namespace test {
namespace {

INSTANTIATE_TEST_SUITE_P(smoke_PagedAttentionTokenType,
                         PagedAttentionTokenTypeTest,
                         ::testing::Combine(::testing::Values(ElementType::f32, ElementType::f16, ElementType::bf16),
                                            ::testing::Values(32),            // head_size
                                            ::testing::Values(1, 4),          // head_num
                                            ::testing::Values(0, 9),          // sliding_window_size
                                            ::testing::Values(1, 4),          // batch_size
                                            ::testing::Values(10, 100, 500),  // seq_len
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         PagedAttentionTokenTypeTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
