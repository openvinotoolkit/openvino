// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "llm_pass_test_fixture.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/softmax.hpp"

namespace {

using ov::test::npuw::RecordingFactory;

class OptimizeValueTensorsPassTest : public ov::test::npuw::LLMPassTestFixture {
protected:
    static bool any_matmul_has_transpose_b(const std::shared_ptr<ov::Model>& model) {
        for (const auto& op : model->get_ops()) {
            if (auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(op)) {
                if (matmul->get_transpose_b()) {
                    return true;
                }
            }
        }
        return false;
    }
};

// ScaledDotProductAttentionDecomposition replaces every SDPA node in both the
// prefill and generate sub-models when OptimizeValueTensors is enabled.
// Parametrize over sub-model suffix to avoid duplicating the identical test body.
class OptimizeValueTensorsNoSDPATest : public ov::test::npuw::LLMPassTestFixture,
                                       public ::testing::WithParamInterface<std::string> {};

INSTANTIATE_TEST_SUITE_P(SubModels,
                          OptimizeValueTensorsNoSDPATest,
                          ::testing::Values(std::string{"_prefill"}, std::string{"_kv192"}),
                          [](const ::testing::TestParamInfo<std::string>& info) {
                              auto name = info.param;
                              if (!name.empty() && name[0] == '_')
                                  name = name.substr(1);
                              return name;
                          });

TEST_P(OptimizeValueTensorsNoSDPATest, NoSDPAOpsAfterOptimization) {
    const auto& suffix = GetParam();
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_OPTIMIZE_V_TENSORS", "YES"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto* sub = recorder.find_suffix(suffix);
    ASSERT_NE(sub, nullptr) << "Sub-model '" << suffix << "' not found";

    EXPECT_EQ(count_ops<ov::op::v13::ScaledDotProductAttention>(sub->model), 0u);
}

// After SDPA decomposition each layer produces two MatMul ops (QK and SV multiplications).
// With 2 layers the generate model must contain at least 4 MatMul ops.
TEST_F(OptimizeValueTensorsPassTest, GenerateModelHasMatMulFromDecomposedSDPA) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_OPTIMIZE_V_TENSORS", "YES"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto* generate = recorder.find_suffix("_kv192");
    ASSERT_NE(generate, nullptr);

    // 2 layers x 2 MatMuls per SDPA decomposition = at least 4
    EXPECT_GE(count_ops<ov::op::v0::MatMul>(generate->model), 4u);
}

// SDPA decomposition inserts a Softmax node for each layer.
// With 2 layers the generate model must contain at least 2 Softmax ops.
TEST_F(OptimizeValueTensorsPassTest, GenerateModelHasSoftmaxFromDecomposedSDPA) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_OPTIMIZE_V_TENSORS", "YES"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto* generate = recorder.find_suffix("_kv192");
    ASSERT_NE(generate, nullptr);

    // At least one Softmax per attention layer; SDPA decomposition creates v8::Softmax
    EXPECT_GE(count_ops<ov::op::v8::Softmax>(generate->model), 2u);
}

// When OptimizeValueTensors is disabled the ScaledDotProductAttentionDecomposition
// sub-pass is skipped.  DecomposeGQA only targets GroupQueryAttention, so the
// standard test model's SDPA nodes must remain intact.
TEST_F(OptimizeValueTensorsPassTest, SDPAOpsRemainsWhenOptimizationDisabled) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_OPTIMIZE_V_TENSORS", "NO"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto* generate = recorder.find_suffix("_kv192");
    ASSERT_NE(generate, nullptr);

    EXPECT_GT(count_ops<ov::op::v13::ScaledDotProductAttention>(generate->model), 0u);
}

// TransposeValueTensors sets transpose_b=true on the value-multiplication MatMul
// when the optimisation fires.  At least one MatMul in the generate model must
// have transpose_b set after the pass runs.
TEST_F(OptimizeValueTensorsPassTest, AtLeastOneMatMulHasTransposeBSet) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_OPTIMIZE_V_TENSORS", "YES"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto* generate = recorder.find_suffix("_kv192");
    ASSERT_NE(generate, nullptr);

    EXPECT_TRUE(any_matmul_has_transpose_b(generate->model));
}

}  // namespace
