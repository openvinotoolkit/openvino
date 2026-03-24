// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>

#include "llm_pass_test_fixture.hpp"

namespace {

using ov::test::npuw::RecordingFactory;

class ReshapeSlicedHeadToStaticPassTest : public ov::test::npuw::LLMPassTestFixture {};

// Test 1: With SHARED_HEAD=YES and MAX_GENERATION_TOKEN_LEN=8, the lm_head input
// must be fully static and shaped {1, 8, 64} (batch=1, gen_len=8, hidden_size=64).
TEST_F(ReshapeSlicedHeadToStaticPassTest, LMHeadInputIsFullyStatic) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_SHARED_HEAD", "YES"},
                                                      {"NPUW_LLM_MAX_GENERATION_TOKEN_LEN", "8"}},
                                                     recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& lm_head = require_sub_model(recorder, "_lm_head");

    const auto head_input = lm_head.model->input(0);
    EXPECT_TRUE(head_input.get_partial_shape().is_static())
        << "lm_head input shape is not fully static: " << head_input.get_partial_shape();
    EXPECT_EQ(head_input.get_shape(), (ov::Shape{1, 8, 64}));
}

// Test 2 (parametrized): The static sequence-length dimension in the lm_head input must
// equal MAX_GENERATION_TOKEN_LEN.
//
// Note: LLMCompiledModel aligns MAX_GENERATION_TOKEN_LEN to the nearest multiple of 8
// (except for the special value 1).  Only multiples of 8 are used as parameters here so
// that the expected shape matches the value actually stored in the compiled model.
class ReshapeSlicedHeadTokenLenTest : public ov::test::npuw::LLMPassTestFixture,
                                      public ::testing::WithParamInterface<size_t> {};

TEST_P(ReshapeSlicedHeadTokenLenTest, LMHeadInputShapeReflectsMaxGenerationTokenLen) {
    const size_t gen_len = GetParam();

    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(
                        {{"NPUW_LLM_SHARED_HEAD", "YES"},
                         {"NPUW_LLM_MAX_GENERATION_TOKEN_LEN", std::to_string(gen_len)}},
                        recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& lm_head = require_sub_model(recorder, "_lm_head");

    const auto head_input = lm_head.model->input(0);
    ASSERT_TRUE(head_input.get_partial_shape().is_static())
        << "lm_head input shape is not fully static: " << head_input.get_partial_shape();
    EXPECT_EQ(head_input.get_shape(), (ov::Shape{1, gen_len, 64}));
}

INSTANTIATE_TEST_SUITE_P(ShapeSweep,
                         ReshapeSlicedHeadTokenLenTest,
                         ::testing::Values(8u, 16u, 24u));

// Test 3: When SHARED_HEAD=NO the pipeline must not produce an lm_head sub-model at all.
TEST_F(ReshapeSlicedHeadToStaticPassTest, LMHeadNotCreatedWithoutSharedHead) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_SHARED_HEAD", "NO"},
                                                      {"NPUW_LLM_MAX_GENERATION_TOKEN_LEN", "8"}},
                                                     recorder));
    ASSERT_NE(compiled, nullptr);

    const auto* lm_head = recorder.find_suffix("_lm_head");
    EXPECT_EQ(lm_head, nullptr)
        << "lm_head sub-model should not be created when SHARED_HEAD=NO, but it was found";
}

// Test 4: With MAX_GENERATION_TOKEN_LEN=1 the third dimension (hidden_size=64 from the
// model configuration) must remain unchanged in the static lm_head input shape.
TEST_F(ReshapeSlicedHeadToStaticPassTest, LMHeadInputHiddenSizeMatchesModelConfig) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_SHARED_HEAD", "YES"},
                                                      {"NPUW_LLM_MAX_GENERATION_TOKEN_LEN", "1"}},
                                                     recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& lm_head = require_sub_model(recorder, "_lm_head");

    const auto head_input = lm_head.model->input(0);
    ASSERT_TRUE(head_input.get_partial_shape().is_static())
        << "lm_head input shape is not fully static: " << head_input.get_partial_shape();

    const auto shape = head_input.get_shape();
    // batch must be 1, seq_len must equal MAX_GENERATION_TOKEN_LEN=1, hidden_size must be 64.
    ASSERT_EQ(shape.size(), 3u);
    EXPECT_EQ(shape[0], 1u)  << "batch dimension mismatch";
    EXPECT_EQ(shape[1], 1u)  << "sequence-length dimension should equal MAX_GENERATION_TOKEN_LEN=1";
    EXPECT_EQ(shape[2], 64u) << "hidden_size dimension must match the model configuration (hidden_size=64)";
}

}  // namespace
