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

class GPURoPETestQwenVL : public RoPETestQwenVL {
protected:
    void SetUp() override {
        RoPETestQwenVL::SetUp();
        const auto& [element_type, _targetDevice, split_op_type] = this->GetParam();
        if (element_type == ov::element::f16) {
            abs_threshold = 0.015f;
        }
    }
};

TEST_P(GPURoPETestQwenVL, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithType(function, {"RoPE"}, 1);
};

const std::vector<std::string> vit_param = {"VariadicSplit", "Slice", "StridedSlice"};
INSTANTIATE_TEST_SUITE_P(smoke_RoPEQwenVL,
                         GPURoPETestQwenVL,
                         ::testing::Combine(
                            ::testing::Values(ov::element::f16, ov::element::f32),
                            ::testing::Values(ov::test::utils::DEVICE_GPU),
                            ::testing::ValuesIn(vit_param)),
                         GPURoPETestQwenVL::getTestCaseName);

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

INSTANTIATE_TEST_SUITE_P(smoke_RoPETestChatGLM,
                         RoPETestChatGLMHF,
                         ::testing::Combine(
                            ::testing::Values(ov::element::f16, ov::element::f32),
                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         RoPETestChatGLMHF::getTestCaseName);

}  // namespace test
}  // namespace ov
