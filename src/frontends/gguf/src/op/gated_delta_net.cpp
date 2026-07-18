// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <cstdint>
#include <memory>
#include <openvino/op/add.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/exp.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/loop.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

// GGML_OP_GATED_DELTA_NET (qwen3next linear-attention block). Implemented as a recurrent OV Loop
// scan over the sequence, matching ggml's per-token gated delta update. This is the portable
// reference path; the OV core also ships an ov::op::internal::GatedDeltaNet fused op, but it is
// left for a hardware-side optimization once validated against a model (the frontend stays
// conversion-only and device-agnostic).
OutputVector translate_gated_delta_net(const NodeContext& context) {
    num_inputs_check(context, 6, 6);

    auto q = context.get_input(0);
    auto k = context.get_input(1);
    auto v = context.get_input(2);
    auto g = context.get_input(3);
    auto beta = context.get_input(4);
    auto state = context.get_input(5);

    auto v_shape = context.get_input_shape(2).to_shape();  // [B, T, H_v, S_v]
    auto q_shape = context.get_input_shape(0).to_shape();  // [B, T, H_k, S_k]
    auto g_shape = context.get_input_shape(3).to_shape();  // [B, T, H_v, 1 or S_v]

    const int64_t B = v_shape[0];
    const int64_t T = v_shape[1];
    const int64_t H_v = v_shape[2];
    const int64_t S_v = v_shape[3];
    const int64_t H_k = q_shape[2];
    const bool kda = (g_shape[3] == (size_t)S_v);

    const int64_t rq1 = H_v / H_k;  // GQA head repeat factor
    const float scale = 1.0f / std::sqrt((float)S_v);

    auto axis_1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto axis_2 = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});

    // [B, T, H, S] -> [B, H, T, S]
    auto perm_0213 = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 2, 1, 3});
    auto q_t = std::make_shared<ov::op::v1::Transpose>(q, perm_0213);
    auto k_t = std::make_shared<ov::op::v1::Transpose>(k, perm_0213);
    auto v_t = std::make_shared<ov::op::v1::Transpose>(v, perm_0213);
    auto g_t = std::make_shared<ov::op::v1::Transpose>(g, perm_0213);
    auto beta_t = std::make_shared<ov::op::v1::Transpose>(beta, perm_0213);

    // Broadcast Q/K heads to V heads (GQA) if needed.
    ov::Output<ov::Node> q_bh = q_t;
    ov::Output<ov::Node> k_bh = k_t;
    if (rq1 > 1) {
        auto q_unsq = std::make_shared<ov::op::v0::Unsqueeze>(q_t, axis_2);
        auto k_unsq = std::make_shared<ov::op::v0::Unsqueeze>(k_t, axis_2);
        auto bcast_shape = ov::op::v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{1, 1, rq1, 1, 1});
        auto q_bcast = std::make_shared<ov::op::v3::Broadcast>(q_unsq, bcast_shape, ov::op::BroadcastType::BIDIRECTIONAL);
        auto k_bcast = std::make_shared<ov::op::v3::Broadcast>(k_unsq, bcast_shape, ov::op::BroadcastType::BIDIRECTIONAL);
        auto perm_5d = ov::op::v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{0, 2, 1, 3, 4});
        auto q_transposed = std::make_shared<ov::op::v1::Transpose>(q_bcast, perm_5d);
        auto k_transposed = std::make_shared<ov::op::v1::Transpose>(k_bcast, perm_5d);
        auto new_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, T, S_v});
        q_bh = std::make_shared<ov::op::v1::Reshape>(q_transposed, new_shape, false);
        k_bh = std::make_shared<ov::op::v1::Reshape>(k_transposed, new_shape, false);
    }

    // Merge batch and head: [B*H_v, T, last]
    auto merge_bh = [&](ov::Output<ov::Node> x, int64_t last_dim) {
        auto shape = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{B * H_v, T, last_dim});
        return std::make_shared<ov::op::v1::Reshape>(x, shape, false);
    };
    auto q_m = merge_bh(q_bh, S_v);
    auto k_m = merge_bh(k_bh, S_v);
    auto v_m = merge_bh(v_t, S_v);
    auto g_m = merge_bh(g_t, kda ? S_v : 1);
    auto beta_m = merge_bh(beta_t, 1);

    auto state_shape = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{B * H_v, S_v, S_v});
    auto state_m = std::make_shared<ov::op::v1::Reshape>(state, state_shape, false);

    auto scale_const = ov::op::v0::Constant::create(ov::element::f32, {}, std::vector<float>{scale});

    // --- Loop body ---
    auto body_state = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto body_q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto body_k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto body_v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto body_g = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto body_beta = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto body_iter = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});

    auto body_cond_out = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, std::vector<bool>{true});

    auto q_t_cur = std::make_shared<ov::op::v8::Gather>(body_q, body_iter, axis_1);
    auto k_t_cur = std::make_shared<ov::op::v8::Gather>(body_k, body_iter, axis_1);
    auto v_t_cur = std::make_shared<ov::op::v8::Gather>(body_v, body_iter, axis_1);
    auto g_t_cur = std::make_shared<ov::op::v8::Gather>(body_g, body_iter, axis_1);
    auto b_t_cur = std::make_shared<ov::op::v8::Gather>(body_beta, body_iter, axis_1);

    auto q_cur = std::make_shared<ov::op::v0::Squeeze>(q_t_cur, axis_1);
    auto k_cur = std::make_shared<ov::op::v0::Squeeze>(k_t_cur, axis_1);
    auto v_cur = std::make_shared<ov::op::v0::Squeeze>(v_t_cur, axis_1);
    auto g_cur = std::make_shared<ov::op::v0::Squeeze>(g_t_cur, axis_1);
    auto b_cur = std::make_shared<ov::op::v0::Squeeze>(b_t_cur, axis_1);

    auto exp_g = std::make_shared<ov::op::v0::Exp>(g_cur);
    auto exp_g_unsq = std::make_shared<ov::op::v0::Unsqueeze>(exp_g, axis_1);
    auto state_decayed = std::make_shared<ov::op::v1::Multiply>(body_state, exp_g_unsq);

    auto k_col = std::make_shared<ov::op::v0::Unsqueeze>(k_cur, axis_2);
    auto sk = std::make_shared<ov::op::v0::MatMul>(state_decayed, k_col, false, false);
    auto sk_sq = std::make_shared<ov::op::v0::Squeeze>(sk, axis_2);
    auto v_minus_sk = std::make_shared<ov::op::v1::Subtract>(v_cur, sk_sq);
    auto delta = std::make_shared<ov::op::v1::Multiply>(v_minus_sk, b_cur);

    auto delta_col = std::make_shared<ov::op::v0::Unsqueeze>(delta, axis_2);
    auto k_row = std::make_shared<ov::op::v0::Unsqueeze>(k_cur, axis_1);
    auto outer_prod = std::make_shared<ov::op::v0::MatMul>(delta_col, k_row, false, false);
    auto state_updated = std::make_shared<ov::op::v1::Add>(state_decayed, outer_prod);

    auto q_col = std::make_shared<ov::op::v0::Unsqueeze>(q_cur, axis_2);
    auto sq = std::make_shared<ov::op::v0::MatMul>(state_updated, q_col, false, false);
    auto sq_squeezed = std::make_shared<ov::op::v0::Squeeze>(sq, axis_2);
    auto attn_out = std::make_shared<ov::op::v1::Multiply>(sq_squeezed, scale_const);
    auto attn_out_unsq = std::make_shared<ov::op::v0::Unsqueeze>(attn_out, axis_1);

    auto body = std::make_shared<ov::Model>(
        ov::OutputVector{body_cond_out, state_updated, attn_out_unsq},
        ov::ParameterVector{body_iter, body_state, body_q, body_k, body_v, body_g, body_beta});

    auto trip_count = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{T});
    auto exec_cond = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, std::vector<bool>{true});

    auto loop = std::make_shared<ov::op::v5::Loop>(trip_count, exec_cond);
    loop->set_function(body);
    loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{0, 0});
    loop->set_merged_input(body_state, state_m, state_updated);
    loop->set_invariant_input(body_q, q_m);
    loop->set_invariant_input(body_k, k_m);
    loop->set_invariant_input(body_v, v_m);
    loop->set_invariant_input(body_g, g_m);
    loop->set_invariant_input(body_beta, beta_m);

    auto final_state_out = loop->get_iter_value(state_updated, -1);
    auto attn_concat_out = loop->get_concatenated_slices(attn_out_unsq, 0, 1, 1, -1, 1);

    // Pack [attn | state] into ggml's output layout: OV [1, 1, T*B + S_v*B, S_v*H_v].
    auto attn_4d_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, T, S_v});
    auto attn_4d = std::make_shared<ov::op::v1::Reshape>(attn_concat_out, attn_4d_shape, false);
    auto attn_perm = std::make_shared<ov::op::v1::Transpose>(attn_4d, perm_0213);
    auto flat_shape_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{-1});
    auto attn_1d = std::make_shared<ov::op::v1::Reshape>(attn_perm, flat_shape_1d, false);

    auto state_4d_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, S_v, S_v});
    auto state_4d = std::make_shared<ov::op::v1::Reshape>(final_state_out, state_4d_shape, false);
    auto state_1d = std::make_shared<ov::op::v1::Reshape>(state_4d, flat_shape_1d, false);

    auto packed = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{attn_1d, state_1d}, 0);
    auto out_shape =
        ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, T * B + S_v * B, S_v * H_v});
    auto res = std::make_shared<ov::op::v1::Reshape>(packed, out_shape, false);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
