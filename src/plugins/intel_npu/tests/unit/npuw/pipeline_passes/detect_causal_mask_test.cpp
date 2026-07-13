// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "../llm_test_helpers.hpp"
#include "model_builder.hpp"
#include "npuw_transformations/detect_causal_mask.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/openvino.hpp"

using ov::test::npuw::make_sliding_window_mask_gemma4;
using ov::test::npuw::make_sliding_window_mask_phi3;
using ov::test::npuw::make_sliding_window_mask_phi3_legacy;
using ov::test::npuw::ModelBuilder;

namespace {

// ---------------------------------------------------------------------------
// Helper: build a minimal 3-input SDPA model (Q/K/V only, no mask).
// is_causal controls the ScaledDotProductAttention attribute.
// ---------------------------------------------------------------------------
std::shared_ptr<ov::Model> make_sdpa_model(bool is_causal) {
    using namespace ov::op;
    auto q = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 8, 64});
    auto k = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 8, 64});
    auto v = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 8, 64});
    auto sdpa = std::make_shared<v13::ScaledDotProductAttention>(q, k, v, is_causal);
    auto result = std::make_shared<v0::Result>(sdpa->output(0));
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{q, k, v});
}

}  // namespace

// ============================================================================
// True-positive tests — pass must return true
// ============================================================================

// Standard LLM: LessEqual(Range→Unsqueeze, Range→Add(offset)→Unsqueeze)
// Covers LLaMA / GPT / Qwen2 / Qwen3 style (all share the make_causal_bool pattern).
TEST(DetectCausalMaskTest, StandardLLM_IsDetected) {
    auto model = ov::test::npuw::build_llm_test_model();
    ASSERT_NE(model, nullptr);
    EXPECT_TRUE(ov::npuw::DetectCausalMask().run_on_model(model));
}

// GQA variant uses the same causal mask structure.
TEST(DetectCausalMaskTest, GQA_IsDetected) {
    auto model = ov::test::npuw::build_llm_gqa_test_model();
    ASSERT_NE(model, nullptr);
    EXPECT_TRUE(ov::npuw::DetectCausalMask().run_on_model(model));
}

// Boolean causal mask — same LessEqual structure but output dtype is boolean.
// Covers Phi-3 plain-causal-boolean and make_causal_mask_boolean variants.
TEST(DetectCausalMaskTest, BooleanCausal_IsDetected) {
    auto cfg = ov::test::npuw::make_test_model_config();
    cfg.boolean_causal_mask = true;
    ModelBuilder mb;
    auto model = mb.build_llm(cfg);
    ASSERT_NE(model, nullptr);
    EXPECT_TRUE(ov::npuw::DetectCausalMask().run_on_model(model));
}

// Whisper decoder — LessEqual(Range→Unsqueeze×3, cache_pos_Range→Unsqueeze×2)
TEST(DetectCausalMaskTest, WhisperDecoder_IsDetected) {
    auto model = ov::test::npuw::build_whisper_decoder_test_model();
    ASSERT_NE(model, nullptr);
    EXPECT_TRUE(ov::npuw::DetectCausalMask().run_on_model(model));
}

// SDPA(is_causal=true) — Path B: no explicit LessEqual mask, only the op attribute.
TEST(DetectCausalMaskTest, SDPAIsCausalAttribute_IsDetected) {
    auto model = make_sdpa_model(/*is_causal=*/true);
    EXPECT_TRUE(ov::npuw::DetectCausalMask().run_on_model(model));
}

// ============================================================================
// True-negative tests — pass must return false
// ============================================================================

// Full attention — SDPA without mask and is_causal=false.
TEST(DetectCausalMaskTest, FullAttentionSDPA_NotDetected) {
    auto model = make_sdpa_model(/*is_causal=*/false);
    EXPECT_FALSE(ov::npuw::DetectCausalMask().run_on_model(model));
}

// Phi-3 / Gemma-2 / Gemma-3 sliding window mask (Phi3SlidingMaskMatcher pattern).
// The LessEqual for the causal component is combined with Greater via BitwiseAnd
// and must NOT be mistaken for a plain causal mask.
TEST(DetectCausalMaskTest, SlidingWindowPhi3_NotDetected) {
    auto model = ov::test::npuw::build_sliding_window_test_model(512, 0, make_sliding_window_mask_phi3);
    ASSERT_NE(model, nullptr);
    EXPECT_FALSE(ov::npuw::DetectCausalMask().run_on_model(model));
}

// Old Phi-3 / transformers 4.51 style — inverted boolean via BitwiseOr + Greater.
TEST(DetectCausalMaskTest, SlidingWindowPhi3Legacy_NotDetected) {
    auto model = ov::test::npuw::build_sliding_window_test_model(512, 0, make_sliding_window_mask_phi3_legacy);
    ASSERT_NE(model, nullptr);
    EXPECT_FALSE(ov::npuw::DetectCausalMask().run_on_model(model));
}

// Gemma-4 sliding window mask (Gemma4SlidingMaskMatcher pattern).
// Uses Add(Range(0,seq)+past_kv_len) for the Q-side — different from Phi-3.
TEST(DetectCausalMaskTest, SlidingWindowGemma4_NotDetected) {
    auto model = ov::test::npuw::build_sliding_window_test_model(512, 0, make_sliding_window_mask_gemma4);
    ASSERT_NE(model, nullptr);
    EXPECT_FALSE(ov::npuw::DetectCausalMask().run_on_model(model));
}

// Gemma-2 style: alternating sliding + full layers. Still a sliding-window model.
TEST(DetectCausalMaskTest, SlidingWindowGemma2Alternating_NotDetected) {
    // sliding_to_full_ratio=1 → one sliding layer per full layer, as in Gemma-2
    auto model = ov::test::npuw::build_sliding_window_test_model(512, 1, make_sliding_window_mask_phi3, 4);
    ASSERT_NE(model, nullptr);
    EXPECT_FALSE(ov::npuw::DetectCausalMask().run_on_model(model));
}

// Default sliding window mask (empty sliding_mask_fn → make_sliding_window_mask):
// LogicalAnd(causal_LessEqual, Greater(kv_row, q_col - window_size)) + Select.
// is_in_sliding_window_pattern() detects the LogicalAnd + Greater combination.
TEST(DetectCausalMaskTest, SlidingWindowDefault_LogicalAnd_NotDetected) {
    // No explicit sliding_mask_fn → builder uses make_sliding_window_mask which
    // produces LogicalAnd(LessEqual, Greater(...)) — different op than BitwiseAnd
    // but still a sliding-window pattern.
    auto model = ov::test::npuw::build_sliding_window_test_model();
    ASSERT_NE(model, nullptr);
    EXPECT_FALSE(ov::npuw::DetectCausalMask().run_on_model(model));
}

// Whisper decoder with boolean mask output.
// make_whisper_causal_mask with boolean_output=true skips the Select-to-float,
// feeding a bool tensor directly to SDPA — exercises boolean mask handling.
TEST(DetectCausalMaskTest, WhisperDecoder_BooleanMask_IsDetected) {
    auto cfg = ov::test::npuw::make_test_model_config<ov::test::npuw::WhisperConfig>();
    cfg.boolean_causal_mask = true;
    ModelBuilder mb;
    auto model = mb.build_whisper_decoder(cfg);
    ASSERT_NE(model, nullptr);
    EXPECT_TRUE(ov::npuw::DetectCausalMask().run_on_model(model));
}

// Qwen3-style causal mask: LessEqual(Range→Unsqueeze, Add(cache_len, Range→Unsqueeze))
// where the Unsqueeze is INSIDE the Add operand (vs. standard where Unsqueeze is OUTSIDE Add).
// Mirrors prepare_embedding_model.cpp::create_new_mask().
TEST(DetectCausalMaskTest, Qwen3StyleExplicit_IsDetected) {
    using namespace ov::op;
    // Inputs: input_ids [1, seq], attention_mask [1, total_seq]
    auto input_ids   = std::make_shared<v0::Parameter>(ov::element::i64, ov::Shape{1, 4});
    auto attn_mask   = std::make_shared<v0::Parameter>(ov::element::i64, ov::Shape{1, 8});

    auto zero   = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto one    = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto zero_f = v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});
    auto neg_inf = v0::Constant::create(ov::element::f32, ov::Shape{}, {-std::numeric_limits<float>::infinity()});

    // seq_len = ShapeOf(input_ids)[1], total_seq = ShapeOf(attn_mask)[1]
    auto ids_shape  = std::make_shared<v3::ShapeOf>(input_ids, ov::element::i64);
    auto mask_shape = std::make_shared<v3::ShapeOf>(attn_mask,  ov::element::i64);
    auto seq_len    = std::make_shared<v8::Gather>(ids_shape,  one, zero);
    auto total_seq  = std::make_shared<v8::Gather>(mask_shape, one, zero);
    auto cache_len  = std::make_shared<v1::Subtract>(total_seq, seq_len);

    // Qwen3 Q-side: Unsqueeze(Range(0, seq_len), 1) → shape [seq, 1]
    auto q_range  = std::make_shared<v4::Range>(zero, seq_len, one, ov::element::i64);
    auto q_unsq   = std::make_shared<v0::Unsqueeze>(q_range, one);
    // causal_threshold = Add(cache_len, q_unsq)  ← Unsqueeze is INSIDE Add operand
    auto threshold = std::make_shared<v1::Add>(cache_len, q_unsq);

    // K-side: Unsqueeze(Range(0, total_seq), 0) → shape [1, total_seq]
    auto k_range  = std::make_shared<v4::Range>(zero, total_seq, one, ov::element::i64);
    auto k_unsq   = std::make_shared<v0::Unsqueeze>(k_range, zero);

    auto causal   = std::make_shared<v1::LessEqual>(k_unsq, threshold);
    auto mask_f   = std::make_shared<v1::Select>(causal, zero_f, neg_inf);
    auto result   = std::make_shared<v0::Result>(mask_f);
    auto model    = std::make_shared<ov::Model>(
        ov::ResultVector{result},
        ov::ParameterVector{input_ids, attn_mask});

    EXPECT_TRUE(ov::npuw::DetectCausalMask().run_on_model(model));
}

// tril-based causal mask — the pattern produced by both the PyTorch frontend
// (torch.tril) and the ONNX frontend (Trilu op, upper=0).
// Both lower to: LessEqual(Unsqueeze(Range(0,M),0), Unsqueeze(Range(0,N),1))
// followed by Select(mask, input, 0).  No sliding-window And+Greater involved.
TEST(DetectCausalMaskTest, TrilPattern_IsDetected) {
    using namespace ov::op;

    auto zero = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto one  = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto N    = v0::Constant::create(ov::element::i64, ov::Shape{}, {8});
    auto M    = v0::Constant::create(ov::element::i64, ov::Shape{}, {8});

    // horizontal_range = Unsqueeze(Range(0, M, 1), 0)  → [1, M]
    auto h_range = std::make_shared<v4::Range>(zero, M, one, ov::element::i64);
    auto h_unsq  = std::make_shared<v0::Unsqueeze>(h_range, zero);

    // vertical_range = Unsqueeze(Range(0, N, 1), 1)  → [N, 1]
    auto v_range = std::make_shared<v4::Range>(zero, N, one, ov::element::i64);
    auto v_unsq  = std::make_shared<v0::Unsqueeze>(v_range, one);

    // tril mask: LessEqual(horizontal, vertical)
    auto mask    = std::make_shared<v1::LessEqual>(h_unsq, v_unsq);

    // Apply mask via Select (same as both frontends)
    auto input   = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{8, 8});
    auto zero_f  = v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});
    auto out     = std::make_shared<v1::Select>(mask, input, zero_f);
    auto result  = std::make_shared<v0::Result>(out);
    auto model   = std::make_shared<ov::Model>(
        ov::ResultVector{result}, ov::ParameterVector{input});

    EXPECT_TRUE(ov::npuw::DetectCausalMask().run_on_model(model));
}

// tril with non-zero diagonal k — PyTorch/ONNX Trilu(k=1).
// vertical_range = Unsqueeze(Range(k_const, N+k_const, 1), 1).
// Range's start is Constant — traces_to_range returns true on the Range node itself.
TEST(DetectCausalMaskTest, TrilOffDiagonal_IsDetected) {
    using namespace ov::op;

    auto zero   = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto one    = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto k      = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});   // diagonal=1
    auto N      = v0::Constant::create(ov::element::i64, ov::Shape{}, {8});
    auto M      = v0::Constant::create(ov::element::i64, ov::Shape{}, {8});

    auto h_range = std::make_shared<v4::Range>(zero, M, one, ov::element::i64);
    auto h_unsq  = std::make_shared<v0::Unsqueeze>(h_range, zero);

    // vertical_range = Range(k, N+k, 1) → Unsqueeze(1)
    auto Nk      = std::make_shared<v1::Add>(N, k);
    auto v_range = std::make_shared<v4::Range>(k, Nk, one, ov::element::i64);
    auto v_unsq  = std::make_shared<v0::Unsqueeze>(v_range, one);

    auto mask    = std::make_shared<v1::LessEqual>(h_unsq, v_unsq);
    auto input   = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{8, 8});
    auto zero_f  = v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});
    auto out     = std::make_shared<v1::Select>(mask, input, zero_f);
    auto result  = std::make_shared<v0::Result>(out);
    auto model   = std::make_shared<ov::Model>(
        ov::ResultVector{result}, ov::ParameterVector{input});

    EXPECT_TRUE(ov::npuw::DetectCausalMask().run_on_model(model));
}
