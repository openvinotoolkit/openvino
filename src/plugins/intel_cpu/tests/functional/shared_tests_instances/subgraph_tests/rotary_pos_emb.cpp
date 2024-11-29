// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/rotary_pos_emb.hpp"

namespace ov {
namespace test {

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestLlama2StridedSlice,
                         RoPETestLlama2StridedSlice,
                         ::testing::Values(ov::test::utils::DEVICE_CPU),
                         RoPETestLlama2StridedSlice::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestChatGLMStridedSlice,
                         RoPETestChatGLMStridedSlice,
                         ::testing::Values(ov::test::utils::DEVICE_CPU),
                         RoPETestChatGLMStridedSlice::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestQwen7bStridedSlice,
                         RoPETestQwen7bStridedSlice,
                         ::testing::Combine(::testing::Values(true, false),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         RoPETestQwen7bStridedSlice::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestGPTJStridedSlice,
                         RoPETestGPTJStridedSlice,
                         ::testing::Combine(::testing::Values(true, false),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         RoPETestGPTJStridedSlice::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestLlama2Slice,
                         RoPETestLlama2Slice,
                         ::testing::Values(ov::test::utils::DEVICE_CPU),
                         RoPETestLlama2Slice::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestChatGLMSlice,
                         RoPETestChatGLMSlice,
                         ::testing::Values(ov::test::utils::DEVICE_CPU),
                         RoPETestChatGLMSlice::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestQwen7bSlice,
                         RoPETestQwen7bSlice,
                         ::testing::Combine(::testing::Values(true, false),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         RoPETestQwen7bSlice::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestGPTJSlice,
                         RoPETestGPTJSlice,
                         ::testing::Combine(::testing::Values(true, false),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         RoPETestGPTJSlice::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestChatGLM,
                         RoPETestChatGLM2DRoPEStridedSlice,
                         ::testing::Values(ov::test::utils::DEVICE_CPU),
                         RoPETestChatGLM2DRoPEStridedSlice::getTestCaseName);

}  // namespace test
}  // namespace ov
