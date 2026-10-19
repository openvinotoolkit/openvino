// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/optimize_value_tensors.hpp"

#include <gtest/gtest.h>

#include "llm_pass_test_fixture.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/transpose.hpp"

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
                         ::testing::Values(std::string{"_prefill"}, std::string{"_kv"}),
                         [](const ::testing::TestParamInfo<std::string>& info) {
                             auto name = info.param;
                             if (!name.empty() && name[0] == '_')
                                 name = name.substr(1);
                             return name;
                         });

TEST_P(OptimizeValueTensorsNoSDPATest, NoSDPAOpsAfterOptimization) {
    const auto& fragment = GetParam();
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_OPTIMIZE_V_TENSORS", "YES"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& sub = require_sub_model_containing(recorder, fragment);

    EXPECT_EQ(count_ops<ov::op::v13::ScaledDotProductAttention>(sub.model), 0u);
}

// After SDPA decomposition each layer produces two MatMul ops (QK and SV multiplications).
// With 2 layers the generate model must contain at least 4 MatMul ops.
TEST_F(OptimizeValueTensorsPassTest, GenerateModelHasMatMulFromDecomposedSDPA) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_OPTIMIZE_V_TENSORS", "YES"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& generate = require_sub_model_containing(recorder, "_kv");

    // 2 layers x 2 MatMuls per SDPA decomposition = at least 4
    EXPECT_GE(count_ops<ov::op::v0::MatMul>(generate.model), 4u);
}

// SDPA decomposition inserts a Softmax node for each layer.
// With 2 layers the generate model must contain at least 2 Softmax ops.
TEST_F(OptimizeValueTensorsPassTest, GenerateModelHasSoftmaxFromDecomposedSDPA) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_OPTIMIZE_V_TENSORS", "YES"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& generate = require_sub_model_containing(recorder, "_kv");

    // At least one Softmax per attention layer; SDPA decomposition creates v8::Softmax
    EXPECT_GE(count_ops<ov::op::v8::Softmax>(generate.model), 2u);
}

// When OptimizeValueTensors is disabled the ScaledDotProductAttentionDecomposition
// sub-pass is skipped.  DecomposeGQA only targets GroupQueryAttention, so the
// standard test model's SDPA nodes must remain intact.
TEST_F(OptimizeValueTensorsPassTest, SDPAOpsRemainsWhenOptimizationDisabled) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_OPTIMIZE_V_TENSORS", "NO"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& generate = require_sub_model_containing(recorder, "_kv");

    EXPECT_GT(count_ops<ov::op::v13::ScaledDotProductAttention>(generate.model), 0u);
}

// TransposeValueTensors sets transpose_b=true on the value-multiplication MatMul
// when the optimisation fires.  At least one MatMul in the generate model must
// have transpose_b set after the pass runs.
TEST_F(OptimizeValueTensorsPassTest, AtLeastOneMatMulHasTransposeBSet) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_OPTIMIZE_V_TENSORS", "YES"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& generate = require_sub_model_containing(recorder, "_kv");

    EXPECT_TRUE(any_matmul_has_transpose_b(generate.model));
}

// Same as above but on a GQA model (num_kv_heads < num_heads), which exercises
// the TransposeValueTensors_GQA pattern instead of the MHA one.
TEST_F(OptimizeValueTensorsPassTest, AtLeastOneMatMulHasTransposeBSet_GQA) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(ov::test::npuw::build_llm_gqa_test_model(),
                                                     {{"NPUW_LLM_OPTIMIZE_V_TENSORS", "YES"}},
                                                     recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& generate = require_sub_model_containing(recorder, "_kv");

    EXPECT_TRUE(any_matmul_has_transpose_b(generate.model));
}

TEST_F(OptimizeValueTensorsPassTest, DirectValuePatternsDoNotDependOnNodeNames) {
    auto generate_value = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3, 4});
    auto generate_scores = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 5, 3});
    auto generate_softmax = std::make_shared<ov::op::v8::Softmax>(generate_scores);
    auto generate_matmul = std::make_shared<ov::op::v0::MatMul>(generate_softmax, generate_value);
    auto generate_model = std::make_shared<ov::Model>(ov::OutputVector{generate_matmul},
                                                      ov::ParameterVector{generate_value, generate_scores});

    EXPECT_TRUE(ov::npuw::util::OptimizeValueTensors(false, true).run_on_model(generate_model));
    EXPECT_TRUE(generate_matmul->get_transpose_b());
    EXPECT_EQ(generate_value->get_partial_shape(), ov::PartialShape({1, 2, 4, 3}));

    auto prefill_value = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 2, 4});
    auto prefill_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
    auto prefill_transpose = std::make_shared<ov::op::v1::Transpose>(prefill_value, prefill_order);
    auto prefill_scores = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 5, 3});
    auto prefill_softmax = std::make_shared<ov::op::v8::Softmax>(prefill_scores);
    auto prefill_matmul = std::make_shared<ov::op::v0::MatMul>(prefill_softmax, prefill_transpose);
    auto prefill_model = std::make_shared<ov::Model>(ov::OutputVector{prefill_matmul},
                                                     ov::ParameterVector{prefill_value, prefill_scores});

    EXPECT_TRUE(ov::npuw::util::OptimizeValueTensors(true, true).run_on_model(prefill_model));
    EXPECT_TRUE(prefill_matmul->get_transpose_b());
    EXPECT_EQ(prefill_transpose->get_output_partial_shape(0), ov::PartialShape({1, 2, 4, 3}));
}

}  // namespace
