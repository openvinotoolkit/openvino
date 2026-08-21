// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Direct unit tests for ov::npuw::SliceLastTokenPrefill.
//
// The pass is tested against a minimal hand-built model that mirrors the
// last-layer attention pattern:
//
//   Q [B,H,N,D]  ──┐
//   K [B,H,S,D]    ├─ SDPA ─► Transpose ─► Reshape([B,N,hidden])
//   V [B,H,S,D]    │            ─► o_proj MatMul ─► Add ◄── shortcut [B,N,hidden]
//   mask [B,1,N,S] ┘
//
// Testing through the full LLMCompiledModel pipeline is not feasible for
// checking SDPA-level changes because DecomposeGQA runs after this pass and
// replaces all SDPA nodes with MatMul+Softmax patterns.

#include <gtest/gtest.h>

#include "npuw_transformations/slice_last_token_prefill.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/transpose.hpp"

namespace {

// Minimal attention model: SDPA → Transpose → Reshape → o_proj → residual Add.
// batch_dim=0, so seq dim in [B,N,hidden] is 1.
std::shared_ptr<ov::Model> build_attention_model(int64_t seq_len = 128,
                                                 int64_t past_len = 192,
                                                 int64_t num_heads = 4,
                                                 int64_t head_dim = 16,
                                                 int64_t hidden = 64) {
    const size_t B = 1, H = static_cast<size_t>(num_heads), N = static_cast<size_t>(seq_len),
                 S = static_cast<size_t>(past_len), D = static_cast<size_t>(head_dim),
                 hidden_u = static_cast<size_t>(hidden);

    auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{B, H, N, D});
    auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{B, H, S, D});
    auto v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{B, H, S, D});
    auto mask = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{B, 1, N, S});
    auto shortcut = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{B, N, hidden_u});

    // SDPA → [B, H, N, D]
    auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, mask, false);

    // Transpose [B,H,N,D] → [B,N,H,D]
    auto perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 2, 1, 3});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(sdpa, perm);

    // Reshape [B,N,H,D] → [B,N,hidden]
    auto shape_const =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{1LL, seq_len, hidden});
    auto reshape = std::make_shared<ov::op::v1::Reshape>(transpose, shape_const, false);

    // o_proj: [B,N,hidden] × [hidden,hidden]
    auto w = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{hidden_u, hidden_u}, std::vector<float>{1.0f});
    auto o_proj = std::make_shared<ov::op::v0::MatMul>(reshape, w, false, true);

    // Residual Add
    auto residual = std::make_shared<ov::op::v1::Add>(o_proj, shortcut);
    auto result = std::make_shared<ov::op::v0::Result>(residual);

    return std::make_shared<ov::Model>(ov::ResultVector({result}),
                                       ov::ParameterVector({q, k, v, mask, shortcut}),
                                       "attention_model");
}

// Returns the last ScaledDotProductAttention in topological order, or nullptr.
std::shared_ptr<ov::op::v13::ScaledDotProductAttention> find_last_sdpa(const std::shared_ptr<ov::Model>& model) {
    std::shared_ptr<ov::op::v13::ScaledDotProductAttention> last;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(node))
            last = sdpa;
    }
    return last;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: Pass fires on a matching model with K=1 (default).
//   - run_on_model() returns true.
//   - SDPA Q input (port 0) is now a v8::Slice.
//   - SDPA mask input (port 3) is now a v8::Slice.
//   - Model output shape: [1, 1, 64].
// ─────────────────────────────────────────────────────────────────────────────
TEST(SlicePrefillLastTokensTest, PassAppliedK1) {
    auto model = build_attention_model(/*seq_len=*/128);
    const bool applied = ov::npuw::SliceLastTokenPrefill(/*batch_dim=*/0, /*num_last=*/1).run_on_model(model);

    ASSERT_TRUE(applied) << "SliceLastTokenPrefill should have matched";

    const auto sdpa = find_last_sdpa(model);
    ASSERT_NE(sdpa, nullptr);

    // Q input must now be a Slice
    EXPECT_TRUE(ov::is_type<ov::op::v8::Slice>(sdpa->input_value(0).get_node_shared_ptr()))
        << "Q input of SDPA is not a v8::Slice after pass";

    // Mask input must now be a Slice
    ASSERT_GT(sdpa->get_input_size(), 3u);
    EXPECT_TRUE(ov::is_type<ov::op::v8::Slice>(sdpa->input_value(3).get_node_shared_ptr()))
        << "mask input of SDPA is not a v8::Slice after pass";

    // Model output shape: [1, 1, 64]
    const auto out = model->outputs()[0].get_partial_shape();
    ASSERT_TRUE(out.is_static()) << "output is not static: " << out;
    EXPECT_EQ(out.to_shape(), (ov::Shape{1, 1, 64}));
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: K=4 (speculative decode). Output seq dim must equal 4.
// ─────────────────────────────────────────────────────────────────────────────
TEST(SlicePrefillLastTokensTest, PassAppliedK4) {
    auto model = build_attention_model(/*seq_len=*/128);
    const bool applied = ov::npuw::SliceLastTokenPrefill(/*batch_dim=*/0, /*num_last=*/4).run_on_model(model);

    ASSERT_TRUE(applied);

    const auto out = model->outputs()[0].get_partial_shape();
    ASSERT_TRUE(out.is_static());
    EXPECT_EQ(out.to_shape(), (ov::Shape{1, 4, 64}));
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: seq_len <= num_last_tokens — nothing to slice, pass returns false.
// ─────────────────────────────────────────────────────────────────────────────
TEST(SlicePrefillLastTokensTest, PassSkippedWhenSeqLenEqualsNumLast) {
    auto model = build_attention_model(/*seq_len=*/4);
    const bool applied = ov::npuw::SliceLastTokenPrefill(/*batch_dim=*/0, /*num_last=*/4).run_on_model(model);

    EXPECT_FALSE(applied) << "pass should be a no-op when seq_len == num_last_tokens";

    // Model output shape must be unchanged: [1, 4, 64]
    const auto out = model->outputs()[0].get_partial_shape();
    ASSERT_TRUE(out.is_static());
    EXPECT_EQ(out.to_shape(), (ov::Shape{1, 4, 64}));
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: Model with no SDPA — pass returns false without modifying the model.
// ─────────────────────────────────────────────────────────────────────────────
TEST(SlicePrefillLastTokensTest, PassSkippedWhenNoSdpa) {
    // Trivial model: param → result, no SDPA.
    auto p = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 128, 64});
    auto r = std::make_shared<ov::op::v0::Result>(p);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{r}, ov::ParameterVector{p});

    const bool applied = ov::npuw::SliceLastTokenPrefill(0, 1).run_on_model(model);
    EXPECT_FALSE(applied);

    // Shape must be unchanged
    const auto out = model->outputs()[0].get_partial_shape();
    ASSERT_TRUE(out.is_static());
    EXPECT_EQ(out.to_shape(), (ov::Shape{1, 128, 64}));
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: Two SDPA nodes (two-layer model).
//   Only the LAST SDPA Q input must be sliced; the first must be unchanged.
// ─────────────────────────────────────────────────────────────────────────────
TEST(SlicePrefillLastTokensTest, OnlyLastSdpaIsSliced) {
    // Build a two-layer chain: layer0_out feeds layer1 as shortcut.
    const size_t B = 1, H = 4, N = 128, S = 192, D = 16, hidden = 64;

    auto q0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{B, H, N, D});
    auto k0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{B, H, S, D});
    auto v0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{B, H, S, D});
    auto mask0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{B, 1, N, S});
    auto sc0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{B, N, hidden});

    auto sdpa0 = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q0, k0, v0, mask0, false);
    auto perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 2, 1, 3});
    auto tr0 = std::make_shared<ov::op::v1::Transpose>(sdpa0, perm);
    auto shc0 = ov::op::v0::Constant::create(ov::element::i64,
                                             ov::Shape{3},
                                             std::vector<int64_t>{1LL, (int64_t)N, (int64_t)hidden});
    auto rs0 = std::make_shared<ov::op::v1::Reshape>(tr0, shc0, false);
    auto w0 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{hidden, hidden}, std::vector<float>{1.0f});
    auto op0 = std::make_shared<ov::op::v0::MatMul>(rs0, w0, false, true);
    auto add0 = std::make_shared<ov::op::v1::Add>(op0, sc0);  // layer0 residual output

    // Layer 1 — the LAST layer; shortcut comes from add0
    auto q1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{B, H, N, D});
    auto k1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{B, H, S, D});
    auto v1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{B, H, S, D});
    auto mask1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{B, 1, N, S});

    auto sdpa1 = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q1, k1, v1, mask1, false);
    auto tr1 = std::make_shared<ov::op::v1::Transpose>(sdpa1, perm);
    auto shc1 = ov::op::v0::Constant::create(ov::element::i64,
                                             ov::Shape{3},
                                             std::vector<int64_t>{1LL, (int64_t)N, (int64_t)hidden});
    auto rs1 = std::make_shared<ov::op::v1::Reshape>(tr1, shc1, false);
    auto w1 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{hidden, hidden}, std::vector<float>{1.0f});
    auto op1 = std::make_shared<ov::op::v0::MatMul>(rs1, w1, false, true);
    auto add1 = std::make_shared<ov::op::v1::Add>(op1, add0);  // shortcut = layer0 output

    auto result = std::make_shared<ov::op::v0::Result>(add1);
    auto model = std::make_shared<ov::Model>(ov::ResultVector({result}),
                                             ov::ParameterVector({q0, k0, v0, mask0, sc0, q1, k1, v1, mask1}),
                                             "two_layer_model");

    const bool applied = ov::npuw::SliceLastTokenPrefill(0, 1).run_on_model(model);
    ASSERT_TRUE(applied);

    // Collect all SDPAs in topological order
    std::vector<std::shared_ptr<ov::op::v13::ScaledDotProductAttention>> sdpas;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto s = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(node))
            sdpas.push_back(s);
    }
    ASSERT_EQ(sdpas.size(), 2u);

    // First SDPA Q input must NOT be a Slice
    EXPECT_FALSE(ov::is_type<ov::op::v8::Slice>(sdpas[0]->input_value(0).get_node_shared_ptr()))
        << "Layer-0 SDPA Q should not be sliced";

    // Last SDPA Q input must be a Slice
    EXPECT_TRUE(ov::is_type<ov::op::v8::Slice>(sdpas[1]->input_value(0).get_node_shared_ptr()))
        << "Layer-1 (last) SDPA Q must be sliced";

    // Overall output shape: [1, 1, 64]
    const auto out = model->outputs()[0].get_partial_shape();
    ASSERT_TRUE(out.is_static());
    EXPECT_EQ(out.to_shape(), (ov::Shape{1, 1, 64}));
}

}  // namespace
