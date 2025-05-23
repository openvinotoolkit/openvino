// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/rotary_pos_emb.hpp"

namespace ov {
namespace test {

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestFlux,
                         RoPETestFlux,
                         ::testing::Combine(
                            ::testing::Values(ov::element::f16, ov::element::f32),
                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         RoPETestFlux::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestChatGLM,
                         RoPETestChatGLMStridedSlice,
                         ::testing::Combine(
                            ::testing::Values(ov::element::f16, ov::element::f32),
                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         RoPETestChatGLMStridedSlice::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestQwen7b,
                         RoPETestQwen7bStridedSlice,
                         ::testing::Combine(::testing::Values(true, false),
                                            ::testing::Values(ov::element::f16, ov::element::f32),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         RoPETestQwen7bStridedSlice::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestLlama2,
                         RoPETestLlama2StridedSlice,
                         ::testing::Combine(
                            ::testing::Values(ov::element::f32),
                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         RoPETestLlama2StridedSlice::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestRotateHalfWithoutTranspose,
                         RoPETestRotateHalfWithoutTranspose,
                         ::testing::Combine(
                            ::testing::Values(ov::element::f32),
                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         RoPETestRotateHalfWithoutTranspose::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestChatGLM,
                         RoPETestChatGLMSlice,
                         ::testing::Combine(
                            ::testing::Values(ov::element::f16, ov::element::f32),
                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         RoPETestChatGLMSlice::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestQwen7b,
                         RoPETestQwen7bSlice,
                         ::testing::Combine(::testing::Values(true, false),
                                            ::testing::Values(ov::element::f16, ov::element::f32),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         RoPETestQwen7bSlice::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestLlama2,
                         RoPETestLlama2Slice,
                         ::testing::Combine(
                            ::testing::Values(ov::element::f32),
                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         RoPETestLlama2Slice::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestChatGLM,
                         RoPETestChatGLM2DRoPEStridedSlice,
                         ::testing::Combine(
                            ::testing::Values(ov::element::f16, ov::element::f32),
                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         RoPETestChatGLM2DRoPEStridedSlice::getTestCaseName);


}  // namespace test
}  // namespace ov
