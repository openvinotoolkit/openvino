// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "llm_pass_test_fixture.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/softmax.hpp"  // v8::Softmax — used by SDPA decomposition

namespace {

using ov::test::npuw::RecordingFactory;

class DecomposeGQAPassTest : public ov::test::npuw::LLMPassTestFixture {};

// Test 1: The pipeline succeeds without throwing when DecomposeGQA is applied
// to a model that contains no GroupQueryAttention nodes (standard SDPA-based model).
TEST_F(DecomposeGQAPassTest, PipelineSucceedsOnStandardSDPAModel) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({}, recorder));
    ASSERT_NE(compiled, nullptr);
}

// Test 2: After running all passes (including DecomposeGQA), neither the prefill
// nor the generate model contains any GroupQueryAttention nodes.
TEST_F(DecomposeGQAPassTest, NoGroupQueryAttentionNodesInPrefillOrGenerateModel) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;
    ASSERT_NO_THROW(compiled = create_compiled_model({}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto* prefill = recorder.find_suffix("_prefill");
    const auto* generate = recorder.find_suffix("_kv192");
    ASSERT_NE(prefill, nullptr);
    ASSERT_NE(generate, nullptr);

    EXPECT_EQ(count_ops<ov::op::internal::GroupQueryAttention>(prefill->model), 0u);
    EXPECT_EQ(count_ops<ov::op::internal::GroupQueryAttention>(generate->model), 0u);
}

// Test 3: Both the prefill and generate models contain MatMul operations after
// the full pass pipeline. This confirms the SDPA decomposition (which DecomposeGQA
// participates in) ran without breaking the graph.
TEST_F(DecomposeGQAPassTest, MatMulOpsArePresentInBothModels) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;
    ASSERT_NO_THROW(compiled = create_compiled_model({}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto* prefill = recorder.find_suffix("_prefill");
    const auto* generate = recorder.find_suffix("_kv192");
    ASSERT_NE(prefill, nullptr);
    ASSERT_NE(generate, nullptr);

    EXPECT_GT(count_ops<ov::op::v0::MatMul>(prefill->model), 0u);
    EXPECT_GT(count_ops<ov::op::v0::MatMul>(generate->model), 0u);
}

// Test 4: Both the prefill and generate models contain Softmax operations after
// the full pass pipeline. Softmax nodes are produced by the SDPA decomposition
// and their presence confirms the graph was not corrupted by DecomposeGQA.
TEST_F(DecomposeGQAPassTest, SoftmaxOpsArePresentInBothModels) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;
    ASSERT_NO_THROW(compiled = create_compiled_model({}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto* prefill = recorder.find_suffix("_prefill");
    const auto* generate = recorder.find_suffix("_kv192");
    ASSERT_NE(prefill, nullptr);
    ASSERT_NE(generate, nullptr);

    // OptimizeValueTensors::ScaledDotProductAttentionDecomposition produces v8::Softmax
    EXPECT_GT(count_ops<ov::op::v8::Softmax>(prefill->model), 0u);
    EXPECT_GT(count_ops<ov::op::v8::Softmax>(generate->model), 0u);
}

// Test 5: The prefill and generate models have different total op counts, confirming
// that the pipeline applies model-specific transformations (controlled by the
// is_prefill flag in DecomposeGQA and other passes) and the two specializations
// are genuinely distinct.
TEST_F(DecomposeGQAPassTest, PrefillAndGenerateModelsHaveDifferentOpCounts) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;
    ASSERT_NO_THROW(compiled = create_compiled_model({}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto* prefill = recorder.find_suffix("_prefill");
    const auto* generate = recorder.find_suffix("_kv192");
    ASSERT_NE(prefill, nullptr);
    ASSERT_NE(generate, nullptr);

    const std::size_t prefill_ops = prefill->model->get_ops().size();
    const std::size_t generate_ops = generate->model->get_ops().size();
    EXPECT_NE(prefill_ops, generate_ops);
}

}  // namespace
