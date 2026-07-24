// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/detect_causal_mask.hpp"

#include <gtest/gtest.h>

#include "../llm_test_helpers.hpp"
#include "model_builder.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/openvino.hpp"

using ov::npuw::DetectAttentionMask;
using ov::npuw::MaskInfo;
using MaskType = ov::npuw::MaskInfo::MaskType;
using ov::test::npuw::make_sliding_window_mask_gemma4;
using ov::test::npuw::make_sliding_window_mask_phi3;
using ov::test::npuw::make_sliding_window_mask_phi3_legacy;
using ov::test::npuw::ModelBuilder;

namespace {

constexpr int64_t kWindow = 512;

MaskType detect(const std::shared_ptr<ov::Model>& model) {
    DetectAttentionMask pass;
    pass.run_on_model(model);
    return pass.get_mask_info().mask_type;
}

// ---------------------------------------------------------------------------
// Minimal Q/K/V SDPA model (no explicit mask). is_causal drives the op attribute.
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

// ---------------------------------------------------------------------------
// Faithful reproduction of the LLaMA / TinyLlama causal-mask subgraph as seen
// in real openvino_model.xml exports:
//
//   kv_idx = Range(0, kv_len)             -> Unsqueeze x3 -> [1,1,1,-1]
//   q_idx  = Range(past, past+seq)        -> Unsqueeze x3 -> [1,1,-1,1]
//   mask   = LessEqual(kv_idx, q_idx)                        (aten::le)
//   mask   = BitwiseAnd(new_ones, mask)                      (aten::__and__)
//
// new_ones is an all-true boolean constant (HF starts the mask from ones),
// so the BitwiseAnd is an identity — no sliding-window Greater involved.
// ---------------------------------------------------------------------------
std::shared_ptr<ov::Model> build_llama_causal() {
    using namespace ov::op;
    auto seq = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});    // input_ids
    auto amask = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});  // attention_mask

    auto zero = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto one = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});

    auto ids_shape = std::make_shared<v3::ShapeOf>(seq, ov::element::i64);
    auto mask_shape = std::make_shared<v3::ShapeOf>(amask, ov::element::i64);
    auto seq_len = std::make_shared<v8::Gather>(ids_shape, one, zero);
    auto kv_len = std::make_shared<v8::Gather>(mask_shape, one, zero);
    auto past = std::make_shared<v1::Subtract>(kv_len, seq_len);

    // kv_idx: Range(0, kv_len) -> [1,1,1,-1]
    auto kv_range = std::make_shared<v4::Range>(zero, kv_len, one, ov::element::i64);
    auto kv0 = std::make_shared<v0::Unsqueeze>(kv_range, one);
    auto kv1 = std::make_shared<v0::Unsqueeze>(kv0, one);
    auto kv_idx = std::make_shared<v0::Unsqueeze>(kv1, one);

    // q_idx: Range(past, past+seq) -> [1,1,-1,1]
    auto q_stop = std::make_shared<v1::Add>(past, seq_len);
    auto q_range = std::make_shared<v4::Range>(past, q_stop, one, ov::element::i64);
    auto q0 = std::make_shared<v0::Unsqueeze>(q_range, one);
    auto q1 = std::make_shared<v0::Unsqueeze>(q0, one);
    auto q_idx = std::make_shared<v0::Unsqueeze>(q1, one);

    auto le = std::make_shared<v1::LessEqual>(kv_idx, q_idx);
    auto new_ones = v0::Constant::create(ov::element::boolean, ov::Shape{}, {true});
    auto combined = std::make_shared<v13::BitwiseAnd>(new_ones, le);
    auto result = std::make_shared<v0::Result>(combined);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{seq, amask});
}

// ---------------------------------------------------------------------------
// Faithful reproduction of the Phi-3.5 sliding-window subgraph as seen in real
// openvino_model.xml exports:
//
//   kv_row  = Range(0, total)          -> Unsqueeze x3 -> [1,1,1,-1]
//   q_col   = Range(past, total)       -> Unsqueeze x3 -> [1,1,-1,1]
//   bound   = Add(q_col, -window)                          (aten::sub as Add)
//   greater = Greater(kv_row, bound)                       (window: kv > q - W)
//   and_win = BitwiseAnd(attn_bool, greater)
//   causal  = LessEqual(kv_row, q_col)                     (causal: kv <= q)
//   mask    = BitwiseAnd(and_win, causal)
// ---------------------------------------------------------------------------
std::shared_ptr<ov::Model> build_phi3_sliding(int64_t window) {
    using namespace ov::op;
    auto seq = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
    auto amask = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});

    auto zero = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto one = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});

    auto ids_shape = std::make_shared<v3::ShapeOf>(seq, ov::element::i64);
    auto mask_shape = std::make_shared<v3::ShapeOf>(amask, ov::element::i64);
    auto seq_len = std::make_shared<v8::Gather>(ids_shape, one, zero);
    auto total = std::make_shared<v8::Gather>(mask_shape, one, zero);
    auto past = std::make_shared<v1::Subtract>(total, seq_len);

    // kv_row: Range(0, total) -> [1,1,1,-1]
    auto kv_range = std::make_shared<v4::Range>(zero, total, one, ov::element::i64);
    auto kv0 = std::make_shared<v0::Unsqueeze>(kv_range, one);
    auto kv1 = std::make_shared<v0::Unsqueeze>(kv0, one);
    auto kv_row = std::make_shared<v0::Unsqueeze>(kv1, one);

    // q_col: Range(past, total) -> [1,1,-1,1]
    auto q_range = std::make_shared<v4::Range>(past, total, one, ov::element::i64);
    auto q0 = std::make_shared<v0::Unsqueeze>(q_range, one);
    auto q1 = std::make_shared<v0::Unsqueeze>(q0, one);
    auto q_col = std::make_shared<v0::Unsqueeze>(q1, one);

    // window bound: kv > q - window  (constant is negative, applied via Add)
    auto neg_window = v0::Constant::create(ov::element::i64, ov::Shape{}, {-window});
    auto bound = std::make_shared<v1::Add>(q_col, neg_window);
    auto greater = std::make_shared<v1::Greater>(kv_row, bound);

    auto attn_bool = std::make_shared<v0::Convert>(amask, ov::element::boolean);
    auto and_win = std::make_shared<v13::BitwiseAnd>(attn_bool, greater);

    auto causal = std::make_shared<v1::LessEqual>(kv_row, q_col);
    auto mask = std::make_shared<v13::BitwiseAnd>(and_win, causal);
    auto result = std::make_shared<v0::Result>(mask);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{seq, amask});
}

// ---------------------------------------------------------------------------
// Faithful reproduction of the MiniCPM causal-mask subgraph (uses aten::lt =>
// Less instead of aten::le => LessEqual), as seen in real MiniCPM exports:
//
//   kv     = Range(0, total)
//   q_col  = Reshape(Add(Range(0, seq), past), [-1, 1])
//   mask   = Less(kv, q_col)                               (aten::lt)
// ---------------------------------------------------------------------------
std::shared_ptr<ov::Model> build_minicpm_less() {
    using namespace ov::op;
    auto seq = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
    auto amask = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});

    auto zero = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto one = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto col_shape = v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{-1, 1});

    auto ids_shape = std::make_shared<v3::ShapeOf>(seq, ov::element::i64);
    auto mask_shape = std::make_shared<v3::ShapeOf>(amask, ov::element::i64);
    auto seq_len = std::make_shared<v8::Gather>(ids_shape, one, zero);
    auto total = std::make_shared<v8::Gather>(mask_shape, one, zero);
    auto past = std::make_shared<v1::Subtract>(total, seq_len);

    auto kv = std::make_shared<v4::Range>(zero, total, one, ov::element::i64);

    auto q_range = std::make_shared<v4::Range>(zero, seq_len, one, ov::element::i64);
    auto q_abs = std::make_shared<v1::Add>(q_range, past);
    auto q_col = std::make_shared<v1::Reshape>(q_abs, col_shape, false);

    auto mask = std::make_shared<v1::Less>(kv, q_col);
    auto result = std::make_shared<v0::Result>(mask);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{seq, amask});
}

}  // namespace

// ============================================================================
// Causal masks — must report MaskType::Causal
// ============================================================================

// NPUW model builder: LLaMA/Qwen-style LessEqual(Range, Range) causal mask.
TEST(DetectAttentionMaskTest, ModelBuilder_StandardLLM_IsCausal) {
    auto model = ov::test::npuw::build_llm_test_model();
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(detect(model), MaskType::Causal);
}

// NPUW model builder: grouped-query attention shares the same causal structure.
TEST(DetectAttentionMaskTest, ModelBuilder_GQA_IsCausal) {
    auto model = ov::test::npuw::build_llm_gqa_test_model();
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(detect(model), MaskType::Causal);
}

// NPUW model builder: boolean causal mask feeding SDPA directly (Phi-3 plain).
TEST(DetectAttentionMaskTest, ModelBuilder_BooleanCausal_IsCausal) {
    auto cfg = ov::test::npuw::make_test_model_config();
    cfg.boolean_causal_mask = true;
    ModelBuilder mb;
    auto model = mb.build_llm(cfg);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(detect(model), MaskType::Causal);
}

// NPUW model builder: Whisper decoder cache-position causal mask.
TEST(DetectAttentionMaskTest, ModelBuilder_WhisperDecoder_IsCausal) {
    auto model = ov::test::npuw::build_whisper_decoder_test_model();
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(detect(model), MaskType::Causal);
}

// NPUW model builder: Whisper decoder with a boolean mask output.
TEST(DetectAttentionMaskTest, ModelBuilder_WhisperDecoderBoolean_IsCausal) {
    auto cfg = ov::test::npuw::make_test_model_config<ov::test::npuw::WhisperConfig>();
    cfg.boolean_causal_mask = true;
    ModelBuilder mb;
    auto model = mb.build_whisper_decoder(cfg);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(detect(model), MaskType::Causal);
}

// Real pattern: LLaMA / TinyLlama LessEqual + BitwiseAnd(new_ones) chain.
TEST(DetectAttentionMaskTest, RealPattern_LlamaCausal_IsCausal) {
    EXPECT_EQ(detect(build_llama_causal()), MaskType::Causal);
}

// Real pattern: MiniCPM uses Less (aten::lt) instead of LessEqual (aten::le).
TEST(DetectAttentionMaskTest, RealPattern_MiniCPMLess_IsCausal) {
    EXPECT_EQ(detect(build_minicpm_less()), MaskType::Causal);
}

// Real pattern: Qwen3 embedding mask — LessEqual(kv, Add(cache_len, q)).
TEST(DetectAttentionMaskTest, RealPattern_Qwen3_IsCausal) {
    using namespace ov::op;
    auto input_ids = std::make_shared<v0::Parameter>(ov::element::i64, ov::Shape{1, 4});
    auto attn_mask = std::make_shared<v0::Parameter>(ov::element::i64, ov::Shape{1, 8});

    auto zero = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto one = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto zero_f = v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});
    auto neg_inf = v0::Constant::create(ov::element::f32, ov::Shape{}, {-std::numeric_limits<float>::infinity()});

    auto ids_shape = std::make_shared<v3::ShapeOf>(input_ids, ov::element::i64);
    auto mask_shape = std::make_shared<v3::ShapeOf>(attn_mask, ov::element::i64);
    auto seq_len = std::make_shared<v8::Gather>(ids_shape, one, zero);
    auto total_seq = std::make_shared<v8::Gather>(mask_shape, one, zero);
    auto cache_len = std::make_shared<v1::Subtract>(total_seq, seq_len);

    auto q_range = std::make_shared<v4::Range>(zero, seq_len, one, ov::element::i64);
    auto q_unsq = std::make_shared<v0::Unsqueeze>(q_range, one);
    auto threshold = std::make_shared<v1::Add>(cache_len, q_unsq);  // Unsqueeze INSIDE the Add

    auto k_range = std::make_shared<v4::Range>(zero, total_seq, one, ov::element::i64);
    auto k_unsq = std::make_shared<v0::Unsqueeze>(k_range, zero);

    auto causal = std::make_shared<v1::LessEqual>(k_unsq, threshold);
    auto mask_f = std::make_shared<v1::Select>(causal, zero_f, neg_inf);
    auto result = std::make_shared<v0::Result>(mask_f);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input_ids, attn_mask});

    EXPECT_EQ(detect(model), MaskType::Causal);
}

// Real pattern: torch.tril / ONNX Trilu — LessEqual(Range unsq0, Range unsq1).
TEST(DetectAttentionMaskTest, RealPattern_Tril_IsCausal) {
    using namespace ov::op;
    auto zero = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto one = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto N = v0::Constant::create(ov::element::i64, ov::Shape{}, {8});

    auto h_range = std::make_shared<v4::Range>(zero, N, one, ov::element::i64);
    auto h_unsq = std::make_shared<v0::Unsqueeze>(h_range, zero);
    auto v_range = std::make_shared<v4::Range>(zero, N, one, ov::element::i64);
    auto v_unsq = std::make_shared<v0::Unsqueeze>(v_range, one);

    auto mask = std::make_shared<v1::LessEqual>(h_unsq, v_unsq);
    auto input = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{8, 8});
    auto zero_f = v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});
    auto out = std::make_shared<v1::Select>(mask, input, zero_f);
    auto result = std::make_shared<v0::Result>(out);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});

    EXPECT_EQ(detect(model), MaskType::Causal);
}

// Real pattern: torch.tril with non-zero diagonal — Range(k, N+k).
TEST(DetectAttentionMaskTest, RealPattern_TrilOffDiagonal_IsCausal) {
    using namespace ov::op;
    auto zero = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto one = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto k = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto N = v0::Constant::create(ov::element::i64, ov::Shape{}, {8});

    auto h_range = std::make_shared<v4::Range>(zero, N, one, ov::element::i64);
    auto h_unsq = std::make_shared<v0::Unsqueeze>(h_range, zero);
    auto Nk = std::make_shared<v1::Add>(N, k);
    auto v_range = std::make_shared<v4::Range>(k, Nk, one, ov::element::i64);
    auto v_unsq = std::make_shared<v0::Unsqueeze>(v_range, one);

    auto mask = std::make_shared<v1::LessEqual>(h_unsq, v_unsq);
    auto input = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{8, 8});
    auto zero_f = v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});
    auto out = std::make_shared<v1::Select>(mask, input, zero_f);
    auto result = std::make_shared<v0::Result>(out);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});

    EXPECT_EQ(detect(model), MaskType::Causal);
}

// SDPA is_causal=true — no explicit mask, causality via the op attribute.
TEST(DetectAttentionMaskTest, SDPAIsCausalAttribute_IsCausal) {
    EXPECT_EQ(detect(make_sdpa_model(/*is_causal=*/true)), MaskType::Causal);
}

// ============================================================================
// Sliding-window masks — must report MaskType::SlidingWindow + window size
// ============================================================================

// NPUW model builder: Phi-3 / Gemma-2 / Gemma-3 sliding-window mask.
TEST(DetectAttentionMaskTest, ModelBuilder_SlidingWindowPhi3_IsSlidingWindow) {
    auto model = ov::test::npuw::build_sliding_window_test_model(kWindow, 0, make_sliding_window_mask_phi3);
    ASSERT_NE(model, nullptr);
    DetectAttentionMask pass;
    pass.run_on_model(model);
    const auto& info = pass.get_mask_info();
    EXPECT_EQ(info.mask_type, MaskInfo::MaskType::SlidingWindow);
    EXPECT_EQ(info.window_size, kWindow);
}

// NPUW model builder: old Phi-3 (transformers 4.51) inverted BitwiseOr pattern.
TEST(DetectAttentionMaskTest, ModelBuilder_SlidingWindowPhi3Legacy_IsSlidingWindow) {
    auto model = ov::test::npuw::build_sliding_window_test_model(kWindow, 0, make_sliding_window_mask_phi3_legacy);
    ASSERT_NE(model, nullptr);
    DetectAttentionMask pass;
    pass.run_on_model(model);
    const auto& info = pass.get_mask_info();
    EXPECT_EQ(info.mask_type, MaskInfo::MaskType::SlidingWindow);
    EXPECT_EQ(info.window_size, kWindow);
}

// NPUW model builder: Gemma-4 sliding-window mask (Add-based cache position).
TEST(DetectAttentionMaskTest, ModelBuilder_SlidingWindowGemma4_IsSlidingWindow) {
    auto model = ov::test::npuw::build_sliding_window_test_model(kWindow, 0, make_sliding_window_mask_gemma4);
    ASSERT_NE(model, nullptr);
    DetectAttentionMask pass;
    pass.run_on_model(model);
    const auto& info = pass.get_mask_info();
    EXPECT_EQ(info.mask_type, MaskInfo::MaskType::SlidingWindow);
    EXPECT_EQ(info.window_size, kWindow);
}

// NPUW model builder: Gemma-2 alternating sliding + full layers.
TEST(DetectAttentionMaskTest, ModelBuilder_SlidingWindowGemma2Alternating_IsSlidingWindow) {
    auto model = ov::test::npuw::build_sliding_window_test_model(kWindow, 1, make_sliding_window_mask_phi3, 4);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(detect(model), MaskInfo::MaskType::SlidingWindow);
}

// NPUW model builder: default float SWA — LogicalAnd(LessEqual, Greater).
TEST(DetectAttentionMaskTest, ModelBuilder_SlidingWindowDefault_IsSlidingWindow) {
    auto model = ov::test::npuw::build_sliding_window_test_model();
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(detect(model), MaskInfo::MaskType::SlidingWindow);
}

// Real pattern: Phi-3.5 Greater(kv, q - W) & LessEqual(kv, q) with a known window.
TEST(DetectAttentionMaskTest, RealPattern_Phi3Sliding_IsSlidingWindowWithSize) {
    const int64_t window = 2047;
    DetectAttentionMask pass;
    pass.run_on_model(build_phi3_sliding(window));
    const auto& info = pass.get_mask_info();
    EXPECT_EQ(info.mask_type, MaskInfo::MaskType::SlidingWindow);
    EXPECT_EQ(info.window_size, window);
}

// ============================================================================
// Unknown / unmasked — must report MaskType::Unknown
// ============================================================================

// Full attention — SDPA with no mask and is_causal=false.
TEST(DetectAttentionMaskTest, FullAttentionSDPA_IsUnknown) {
    EXPECT_EQ(detect(make_sdpa_model(/*is_causal=*/false)),
              MaskInfo::MaskType::Unknown);
}
