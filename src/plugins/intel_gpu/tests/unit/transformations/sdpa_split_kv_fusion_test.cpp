// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Unit tests for SDPASplitKVFusion: the GPU plugin pass that fuses the Gemma-style split-attention
// decode sub-graph (cache + new K/V attended separately, then summed) into a single split-KV
// op::SDPA. Covers the plain-decode and GQA-folded positive cases, plus the multi-token and
// dynamic-shape cases that must be left unfused.

#include <gtest/gtest.h>

#include <memory>

#include <openvino/core/model.hpp>
#include <openvino/pass/manager.hpp>

#include "intel_gpu/op/sdpa.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/variadic_split.hpp"
#include "plugin/transformations/sdpa_split_kv_fusion.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace {
namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v8 = ov::op::v8;

// Dimensions for the non-GQA path: q_heads == kv_heads == H, so Q reaches the QK matmul already in
// canonical [B, H, S_q, D] layout (no fold Reshape) and the fusion takes its no-un-fold branch.
struct SplitKVDims {
    int64_t B = 1;
    int64_t H = 8;
    int64_t D = 64;
    int64_t S_q = 1;      // decode
    int64_t S_cache = 128;
    int64_t S_new = 1;
};

// Build the split-attention decode sub-graph that SDPASplitKVFusion matches, with the default
// contiguous [B,H,S,D] K/V layout (qk transpose_b=true, attn transpose_b=false):
//
//   qk_cache = MatMul(Q, K_cache, transpose_b)   qk_new = MatMul(Q, K_new, transpose_b)
//   scores   = Concat(qk_cache, qk_new, -1)
//   probs    = Softmax(scores + mask, -1)
//   [pc, pn] = VariadicSplit(probs, -1, {S_cache, S_new})
//   out      = MatMul(pc, V_cache) + MatMul(pn, V_new)
std::shared_ptr<ov::Model> build_split_kv_decode_model(const SplitKVDims& d, bool dynamic_cache = false) {
    const auto et = ov::element::f32;
    const auto cache = dynamic_cache ? ov::Dimension::dynamic() : ov::Dimension(d.S_cache);
    const auto mask_kv = dynamic_cache ? ov::Dimension::dynamic() : ov::Dimension(d.S_cache + d.S_new);

    auto Q = std::make_shared<v0::Parameter>(et, ov::PartialShape{d.B, d.H, d.S_q, d.D});
    auto Kc = std::make_shared<v0::Parameter>(et, ov::PartialShape{d.B, d.H, cache, d.D});
    auto Kn = std::make_shared<v0::Parameter>(et, ov::PartialShape{d.B, d.H, d.S_new, d.D});
    auto Vc = std::make_shared<v0::Parameter>(et, ov::PartialShape{d.B, d.H, cache, d.D});
    auto Vn = std::make_shared<v0::Parameter>(et, ov::PartialShape{d.B, d.H, d.S_new, d.D});
    auto M = std::make_shared<v0::Parameter>(et, ov::PartialShape{d.B, 1, d.S_q, mask_kv});

    auto qk_cache = std::make_shared<v0::MatMul>(Q, Kc, /*transpose_a*/ false, /*transpose_b*/ true);
    auto qk_new = std::make_shared<v0::MatMul>(Q, Kn, /*transpose_a*/ false, /*transpose_b*/ true);

    auto scores = std::make_shared<v0::Concat>(ov::OutputVector{qk_cache, qk_new}, -1);
    auto masked = std::make_shared<v1::Add>(scores, M);
    auto probs = std::make_shared<v8::Softmax>(masked, -1);

    auto split_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
    auto split_sizes = v0::Constant::create(ov::element::i64, ov::Shape{2}, {dynamic_cache ? -1 : d.S_cache, d.S_new});
    auto split = std::make_shared<v1::VariadicSplit>(probs, split_axis, split_sizes);

    auto attn_cache = std::make_shared<v0::MatMul>(split->output(0), Vc, false, false);
    auto attn_new = std::make_shared<v0::MatMul>(split->output(1), Vn, false, false);
    auto out = std::make_shared<v1::Add>(attn_cache, attn_new);

    return std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{Q, Kc, Kn, Vc, Vn, M});
}

// Build a GQA decode sub-graph where the export folds the group (q_heads / kv_heads) into Q's
// query axis: Q is projected as [B, S_q, q_heads, D], then a Reshape folds it to the QK operand
// [B, kv_heads, group*S_q, D] (with S_q == 1, group*S_q == group). The matcher's QK operands then
// batch over kv_heads, and the fusion must un-fold Q (and slice the group-replicated mask) back to
// the canonical [B, q_heads, S_q, D], re-folding the SDPA output to the matched Add's layout.
std::shared_ptr<ov::Model> build_gqa_fold_decode_model(int64_t q_heads, int64_t kv_heads) {
    const auto et = ov::element::f32;
    const int64_t B = 1, D = 64, S_q = 1, S_cache = 128, S_new = 1;
    const int64_t group = q_heads / kv_heads;

    // Unfolded Q projection [B, S_q, q_heads, D]; reshape folds it to [B, kv_heads, group*S_q, D].
    auto Qproj = std::make_shared<v0::Parameter>(et, ov::PartialShape{B, S_q, q_heads, D});
    auto fold = v0::Constant::create(ov::element::i64, ov::Shape{4}, {B, kv_heads, group * S_q, D});
    auto Q = std::make_shared<v1::Reshape>(Qproj, fold, false);

    auto Kc = std::make_shared<v0::Parameter>(et, ov::PartialShape{B, kv_heads, S_cache, D});
    auto Kn = std::make_shared<v0::Parameter>(et, ov::PartialShape{B, kv_heads, S_new, D});
    auto Vc = std::make_shared<v0::Parameter>(et, ov::PartialShape{B, kv_heads, S_cache, D});
    auto Vn = std::make_shared<v0::Parameter>(et, ov::PartialShape{B, kv_heads, S_new, D});
    // Mask is group-replicated along the query axis: [B, 1, group*S_q, S_cache + S_new].
    auto M = std::make_shared<v0::Parameter>(et, ov::PartialShape{B, 1, group * S_q, S_cache + S_new});

    auto qk_cache = std::make_shared<v0::MatMul>(Q, Kc, false, true);
    auto qk_new = std::make_shared<v0::MatMul>(Q, Kn, false, true);
    auto scores = std::make_shared<v0::Concat>(ov::OutputVector{qk_cache, qk_new}, -1);
    auto masked = std::make_shared<v1::Add>(scores, M);
    auto probs = std::make_shared<v8::Softmax>(masked, -1);
    auto split_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
    auto split_sizes = v0::Constant::create(ov::element::i64, ov::Shape{2}, {S_cache, S_new});
    auto split = std::make_shared<v1::VariadicSplit>(probs, split_axis, split_sizes);
    auto attn_cache = std::make_shared<v0::MatMul>(split->output(0), Vc, false, false);
    auto attn_new = std::make_shared<v0::MatMul>(split->output(1), Vn, false, false);
    auto out = std::make_shared<v1::Add>(attn_cache, attn_new);

    return std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{Qproj, Kc, Kn, Vc, Vn, M});
}

std::shared_ptr<op::SDPA> find_split_kv_sdpa(const std::shared_ptr<ov::Model>& model) {
    for (const auto& op : model->get_ops()) {
        if (auto sdpa = ov::as_type_ptr<op::SDPA>(op)) {
            if (sdpa->get_split_kv()) {
                return sdpa;
            }
        }
    }
    return nullptr;
}

std::shared_ptr<op::SDPA> run_fusion(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager;
    manager.register_pass<SDPASplitKVFusion>();
    manager.run_passes(model);
    return find_split_kv_sdpa(model);
}
}  // namespace

// Decode (S_q == 1), static shapes, default layout: the sub-graph fuses into a single split-KV
// op::SDPA with the 7-input layout [Q, K_cache, V_cache, mask, K_new, V_new, kv_len].
TEST(SDPASplitKVFusionTest, Decode) {
    auto model = build_split_kv_decode_model(SplitKVDims{});

    auto sdpa = run_fusion(model);
    ASSERT_NE(sdpa, nullptr) << "split-KV decode sub-graph was not fused into op::SDPA(split_kv=true)";
    EXPECT_EQ(sdpa->get_input_size(), 7u);  // [Q, K_cache, V_cache, mask, K_new, V_new, kv_len]
    EXPECT_EQ(sdpa->get_output_partial_shape(0), (ov::PartialShape{1, 8, 1, 64}));
}

// GQA decode: the export folds the group into Q's query axis. The fusion must un-fold Q to the
// canonical [B, q_heads, S_q, D] (so the op batches over q_heads, not kv_heads), and re-fold the
// SDPA output back to the matched Add's folded layout [B, kv_heads, group*S_q, D] so downstream
// shapes are preserved. q_heads=32, kv_heads=8 (group=4).
TEST(SDPASplitKVFusionTest, GqaFoldDecode) {
    auto model = build_gqa_fold_decode_model(/*q_heads=*/32, /*kv_heads=*/8);

    auto sdpa = run_fusion(model);
    ASSERT_NE(sdpa, nullptr) << "GQA-folded split-KV decode sub-graph was not fused";
    EXPECT_EQ(sdpa->get_input_size(), 7u);
    // The op carries the canonical (un-folded) Q layout [B, q_heads, S_q, D].
    EXPECT_EQ(sdpa->get_output_partial_shape(0), (ov::PartialShape{1, 32, 1, 64}));
    // Its single consumer is the output re-fold Reshape back to the folded [B, kv_heads, group, D].
    const auto consumers = sdpa->get_output_target_inputs(0);
    ASSERT_EQ(consumers.size(), 1u);
    auto refold = ov::as_type_ptr<ov::op::v1::Reshape>(consumers.begin()->get_node()->shared_from_this());
    ASSERT_NE(refold, nullptr) << "expected an output re-fold Reshape after the un-folded SDPA";
    EXPECT_EQ(refold->get_output_partial_shape(0), (ov::PartialShape{1, 8, 4, 64}));
}

// Prefill / multi-token chunk (S_q > 1, S_new > 1) is out of scope for the decode-only kernel and
// must be left unfused.
TEST(SDPASplitKVFusionTest, MultiTokenNoFusion) {
    SplitKVDims d;
    d.S_q = 16;
    d.S_new = 16;
    auto model = build_split_kv_decode_model(d);

    EXPECT_EQ(run_fusion(model), nullptr) << "multi-token chunk must not fuse into split-KV SDPA";
}

// Dynamic shapes have no split-KV kernel implementation, so the match must bail.
TEST(SDPASplitKVFusionTest, DynamicNoFusion) {
    auto model = build_split_kv_decode_model(SplitKVDims{}, /*dynamic_cache=*/true);

    EXPECT_EQ(run_fusion(model), nullptr) << "dynamic shapes must not fuse into split-KV SDPA";
}
