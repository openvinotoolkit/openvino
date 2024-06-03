// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/rotary_pos_emb.hpp"

namespace ov {
namespace test {

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestLlama2,
                         RoPETestLlama2,
                         ::testing::Values(ov::test::utils::DEVICE_CPU),
                         RoPETestLlama2::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestChatGLM,
                         RoPETestChatGLM,
                         ::testing::Values(ov::test::utils::DEVICE_CPU),
                         RoPETestChatGLM::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestQwen7b,
                         RoPETestQwen7b,
                         ::testing::Combine(::testing::Values(true, false),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         RoPETestQwen7b::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestGPTJ,
                         RoPETestGPTJ,
                         ::testing::Combine(::testing::Values(true, false),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         RoPETestGPTJ::getTestCaseName);
}  // namespace test
}  // namespace ov
