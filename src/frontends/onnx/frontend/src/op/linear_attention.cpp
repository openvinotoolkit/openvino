// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <string>

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/gated_delta_net.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace detail {
namespace {

// (B, T, H * D) -> (B, T, H, D)
ov::Output<ov::Node> split_heads(const ov::Output<ov::Node>& input, int64_t num_heads, int64_t head_size) {
    const auto pattern =
        v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, num_heads, head_size});
    return std::make_shared<v1::Reshape>(input, pattern, true);
}

// Interleave-repeat every head `group` times, matching the ONNX reference's np.repeat(..., axis=head_axis).
// (B, T, H, D) -> (B, T, H * group, D)
ov::Output<ov::Node> repeat_heads_4d(const ov::Output<ov::Node>& input, int64_t group, int64_t head_size) {
    if (group == 1) {
        return input;
    }
    const auto axis_3 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {3});
    const auto unsqueezed = std::make_shared<v0::Unsqueeze>(input, axis_3);
    const auto repeats = v0::Constant::create(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{1, 1, 1, group, 1});
    const auto tiled = std::make_shared<v0::Tile>(unsqueezed, repeats);
    const auto pattern = v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, -1, head_size});
    return std::make_shared<v1::Reshape>(tiled, pattern, true);
}

// (B, T, H) -> (B, T, H * group)
ov::Output<ov::Node> repeat_heads_3d(const ov::Output<ov::Node>& input, int64_t group) {
    if (group == 1) {
        return input;
    }
    const auto axis_3 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {3});
    const auto unsqueezed = std::make_shared<v0::Unsqueeze>(input, axis_3);
    const auto repeats = v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, 1, 1, group});
    const auto tiled = std::make_shared<v0::Tile>(unsqueezed, repeats);
    const auto pattern = v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 0, -1});
    return std::make_shared<v1::Reshape>(tiled, pattern, true);
}

// (B, H, d_k, d_v) -> (B, H * group, d_k, d_v)
ov::Output<ov::Node> repeat_state_heads(const ov::Output<ov::Node>& state,
                                        int64_t group,
                                        int64_t d_k,
                                        int64_t d_v) {
    if (group == 1) {
        return state;
    }
    const auto axis_2 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    const auto unsqueezed = std::make_shared<v0::Unsqueeze>(state, axis_2);
    const auto repeats = v0::Constant::create(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{1, 1, group, 1, 1});
    const auto tiled = std::make_shared<v0::Tile>(unsqueezed, repeats);
    const auto pattern = v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, -1, d_k, d_v});
    return std::make_shared<v1::Reshape>(tiled, pattern, true);
}

ov::Output<ov::Node> convert_to(const ov::Output<ov::Node>& input, const ov::element::Type& type) {
    if (input.get_element_type() == type) {
        return input;
    }
    return std::make_shared<v0::Convert>(input, type);
}

struct RecurrenceIO {
    ov::Output<ov::Node> output;         // (B*H, T, d_v)
    ov::Output<ov::Node> present_state;  // (B*H, d_k, d_v)
};

// Sequential linear-attention recurrence built as an OV Loop, all math in f32. Inputs are merged
// per (batch*head) and already GQA-expanded to H = q_num_heads. Implements, per time step t:
//   if gating: S *= exp(g_t)            (g_t per key-dim, broadcast over d_v)
//   if delta:  v_t = beta_t * (v_t - S^T k_t)
//   S += k_t (x) v_t ;  o_t = scale * q_t^T S
RecurrenceIO build_recurrence_loop(const ov::Output<ov::Node>& q_m,     // (BH, T, d_k)
                                   const ov::Output<ov::Node>& k_m,     // (BH, T, d_k)
                                   const ov::Output<ov::Node>& v_m,     // (BH, T, d_v)
                                   const ov::Output<ov::Node>& g_m,     // (BH, T, d_k) or empty
                                   const ov::Output<ov::Node>& beta_m,  // (BH, T, 1) or empty
                                   const ov::Output<ov::Node>& state_m,  // (BH, d_k, d_v)
                                   float scale_value,
                                   bool gating,
                                   bool delta) {
    const auto f32 = ov::element::f32;
    const auto axis_0 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    const auto axis_1 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    const auto axis_2 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});

    auto body_iter = std::make_shared<v0::Parameter>(ov::element::i64, ov::Shape{1});
    auto body_state = std::make_shared<v0::Parameter>(f32, ov::PartialShape::dynamic(3));
    auto body_q = std::make_shared<v0::Parameter>(f32, ov::PartialShape::dynamic(3));
    auto body_k = std::make_shared<v0::Parameter>(f32, ov::PartialShape::dynamic(3));
    auto body_v = std::make_shared<v0::Parameter>(f32, ov::PartialShape::dynamic(3));
    std::shared_ptr<v0::Parameter> body_g, body_beta;

    auto gather_t = [&](const std::shared_ptr<v0::Parameter>& p) -> ov::Output<ov::Node> {
        auto g = std::make_shared<v8::Gather>(p, body_iter, axis_1);  // (BH,1,·)
        return std::make_shared<v0::Squeeze>(g, axis_1);              // (BH,·)
    };
    auto q_cur = gather_t(body_q);
    auto k_cur = gather_t(body_k);
    auto v_cur = gather_t(body_v);

    ov::Output<ov::Node> state = body_state;
    if (gating) {
        body_g = std::make_shared<v0::Parameter>(f32, ov::PartialShape::dynamic(3));
        auto g_cur = gather_t(body_g);                                    // (BH,d_k)
        auto exp_g = std::make_shared<v0::Exp>(g_cur);
        auto exp_col = std::make_shared<v0::Unsqueeze>(exp_g, axis_2);    // (BH,d_k,1)
        state = std::make_shared<v1::Multiply>(state, exp_col);
    }
    ov::Output<ov::Node> v_eff = v_cur;
    if (delta) {
        body_beta = std::make_shared<v0::Parameter>(f32, ov::PartialShape::dynamic(3));
        auto beta_cur = gather_t(body_beta);                             // (BH,1)
        auto k_row = std::make_shared<v0::Unsqueeze>(k_cur, axis_1);      // (BH,1,d_k)
        auto retrieved = std::make_shared<v0::MatMul>(k_row, state, false, false);  // (BH,1,d_v)
        auto retrieved_sq = std::make_shared<v0::Squeeze>(retrieved, axis_1);       // (BH,d_v)
        auto diff = std::make_shared<v1::Subtract>(v_cur, retrieved_sq);
        v_eff = std::make_shared<v1::Multiply>(diff, beta_cur);
    }
    auto k_col = std::make_shared<v0::Unsqueeze>(k_cur, axis_2);          // (BH,d_k,1)
    auto v_row = std::make_shared<v0::Unsqueeze>(v_eff, axis_1);          // (BH,1,d_v)
    auto outer = std::make_shared<v0::MatMul>(k_col, v_row, false, false);  // (BH,d_k,d_v)
    auto state_updated = std::make_shared<v1::Add>(state, outer);

    auto q_row = std::make_shared<v0::Unsqueeze>(q_cur, axis_1);          // (BH,1,d_k)
    auto read = std::make_shared<v0::MatMul>(q_row, state_updated, false, false);  // (BH,1,d_v)
    auto read_sq = std::make_shared<v0::Squeeze>(read, axis_1);          // (BH,d_v)
    auto scale_c = v0::Constant::create(f32, ov::Shape{}, {scale_value});
    auto o_scaled = std::make_shared<v1::Multiply>(read_sq, scale_c);
    auto o_step = std::make_shared<v0::Unsqueeze>(o_scaled, axis_1);      // (BH,1,d_v)

    auto cond = v0::Constant::create(ov::element::boolean, ov::Shape{1}, {true});

    ov::ParameterVector params{body_iter, body_state, body_q, body_k, body_v};
    if (gating) {
        params.push_back(body_g);
    }
    if (delta) {
        params.push_back(body_beta);
    }
    auto body = std::make_shared<ov::Model>(ov::OutputVector{cond, state_updated, o_step}, params);

    auto q_shape = std::make_shared<v3::ShapeOf>(q_m, ov::element::i64);
    auto trip = std::make_shared<v8::Gather>(q_shape, axis_1, axis_0);  // (1,) = T
    auto exec = v0::Constant::create(ov::element::boolean, ov::Shape{1}, {true});

    auto loop = std::make_shared<v5::Loop>(trip, exec);
    loop->set_function(body);
    loop->set_special_body_ports(v5::Loop::SpecialBodyPorts{0, 0});
    loop->set_merged_input(body_state, state_m, state_updated);
    loop->set_invariant_input(body_q, q_m);
    loop->set_invariant_input(body_k, k_m);
    loop->set_invariant_input(body_v, v_m);
    if (gating) {
        loop->set_invariant_input(body_g, g_m);
    }
    if (delta) {
        loop->set_invariant_input(body_beta, beta_m);
    }

    RecurrenceIO result;
    result.present_state = loop->get_iter_value(state_updated, -1);           // (BH,d_k,d_v)
    result.output = loop->get_concatenated_slices(o_step, 0, 1, 1, -1, 1);    // (BH,T,d_v)
    return result;
}

}  // namespace
}  // namespace detail

namespace opset_27 {

// LinearAttention-27. The "delta" rule and the per-head-scalar "gated_delta" rule map directly onto
// the internal ov::op::internal::GatedDeltaNet op (fast path). All other configurations
// ("linear"/"gated" rules and per-key-dim "gated_delta") are lowered to a serializable OV Loop that
// runs the same per-token recurrence with core ops.
ov::OutputVector linear_attention(const ov::frontend::onnx::Node& node) {
    const auto inputs = node.get_ov_inputs();
    CHECK_VALID_NODE(node,
                     inputs.size() >= 3 && inputs.size() <= 6,
                     "LinearAttention expects from 3 to 6 inputs, got: ",
                     inputs.size());

    const auto& query = inputs[0];
    const auto& key = inputs[1];
    const auto& value = inputs[2];
    const bool has_past_state = common::is_input_valid(node, 3);
    const bool has_decay = common::is_input_valid(node, 4);
    const bool has_beta = common::is_input_valid(node, 5);

    const auto q_num_heads = node.get_attribute_value<int64_t>("q_num_heads");
    const auto kv_num_heads = node.get_attribute_value<int64_t>("kv_num_heads");
    const auto scale = node.get_attribute_value<float>("scale", 0.0f);
    const auto update_rule = node.get_attribute_value<std::string>("update_rule", "gated_delta");

    CHECK_VALID_NODE(node,
                     update_rule == "linear" || update_rule == "gated" || update_rule == "delta" ||
                         update_rule == "gated_delta",
                     "LinearAttention: unsupported update_rule '",
                     update_rule,
                     "', expected one of: linear, gated, delta, gated_delta");
    const bool gating = (update_rule == "gated" || update_rule == "gated_delta");
    const bool delta = (update_rule == "delta" || update_rule == "gated_delta");

    CHECK_VALID_NODE(node,
                     q_num_heads > 0 && kv_num_heads > 0 && q_num_heads % kv_num_heads == 0,
                     "LinearAttention: q_num_heads (",
                     q_num_heads,
                     ") must be a positive multiple of kv_num_heads (",
                     kv_num_heads,
                     ")");
    CHECK_VALID_NODE(node,
                     has_decay == gating,
                     "LinearAttention: the decay input is required by the 'gated'/'gated_delta' rules and forbidden "
                     "otherwise");
    CHECK_VALID_NODE(node,
                     has_beta == delta,
                     "LinearAttention: the beta input is required by the 'delta'/'gated_delta' rules and forbidden "
                     "otherwise");

    const auto& query_ps = query.get_partial_shape();
    const auto& value_ps = value.get_partial_shape();
    CHECK_VALID_NODE(node,
                     query_ps.rank().is_static() && query_ps.rank().get_length() == 3 && query_ps[2].is_static(),
                     "LinearAttention: query must be a rank-3 tensor with a static hidden size, got: ",
                     query_ps);
    CHECK_VALID_NODE(node,
                     value_ps.rank().is_static() && value_ps.rank().get_length() == 3 && value_ps[2].is_static(),
                     "LinearAttention: value must be a rank-3 tensor with a static hidden size, got: ",
                     value_ps);
    CHECK_VALID_NODE(node,
                     query_ps[2].get_length() % q_num_heads == 0,
                     "LinearAttention: query hidden size ",
                     query_ps[2].get_length(),
                     " is not divisible by q_num_heads ",
                     q_num_heads);
    CHECK_VALID_NODE(node,
                     value_ps[2].get_length() % kv_num_heads == 0,
                     "LinearAttention: value hidden size ",
                     value_ps[2].get_length(),
                     " is not divisible by kv_num_heads ",
                     kv_num_heads);

    const int64_t d_k = query_ps[2].get_length() / q_num_heads;
    const int64_t d_v = value_ps[2].get_length() / kv_num_heads;
    const int64_t group = q_num_heads / kv_num_heads;

    const auto compute_type = query.get_element_type();
    CHECK_VALID_NODE(node,
                     compute_type.is_real() && compute_type != ov::element::dynamic,
                     "LinearAttention: query element type must be a static floating point type, got: ",
                     compute_type);

    // decay layout: per-head scalar (kv_num_heads) or per-key-dim (kv_num_heads * d_k).
    bool decay_per_head = false;
    if (gating) {
        const auto& decay_ps = inputs[4].get_partial_shape();
        CHECK_VALID_NODE(node,
                         decay_ps.rank().is_static() && decay_ps.rank().get_length() == 3 && decay_ps[2].is_static(),
                         "LinearAttention: decay must be a rank-3 tensor with a static last dimension, got: ",
                         decay_ps);
        const int64_t decay_last = decay_ps[2].get_length();
        CHECK_VALID_NODE(node,
                         decay_last == kv_num_heads || decay_last == kv_num_heads * d_k,
                         "LinearAttention: decay last dimension must be kv_num_heads (",
                         kv_num_heads,
                         ") or kv_num_heads*d_k (",
                         kv_num_heads * d_k,
                         "), got: ",
                         decay_last);
        decay_per_head = (decay_last == kv_num_heads);
    }

    int64_t beta_last_dim = 0;
    if (delta) {
        const auto& beta_ps = inputs[5].get_partial_shape();
        CHECK_VALID_NODE(node,
                         beta_ps.rank().is_static() && beta_ps.rank().get_length() == 3 && beta_ps[2].is_static(),
                         "LinearAttention: beta must be a rank-3 tensor with a static last dimension, got: ",
                         beta_ps);
        beta_last_dim = beta_ps[2].get_length();
        CHECK_VALID_NODE(node,
                         beta_last_dim == kv_num_heads || beta_last_dim == 1,
                         "LinearAttention: beta last dimension must be kv_num_heads (",
                         kv_num_heads,
                         ") or 1, got: ",
                         beta_last_dim);
    }

    const auto i64 = ov::element::i64;
    const auto axis0_1 = v0::Constant::create(i64, ov::Shape{1}, {0});
    const auto query_shape = std::make_shared<v3::ShapeOf>(query, i64);
    const auto batch =
        std::make_shared<v8::Gather>(query_shape, v0::Constant::create(i64, ov::Shape{1}, {0}), axis0_1);

    // ---- Fast path: reuse the internal GatedDeltaNet op (per-head scalar gate + delta correction). ----
    // GatedDeltaNet keeps one state per value head and shares a query/key head across a group of value
    // heads, opposite to the ONNX grouping. Replicating key/value/decay/beta up to q_num_heads makes the
    // conventions agree; the replicated states stay identical, so present_state is reduced back afterwards.
    if (delta && (!gating || decay_per_head)) {
        auto q_4d = detail::split_heads(query, q_num_heads, d_k);
        auto k_4d =
            detail::repeat_heads_4d(detail::split_heads(detail::convert_to(key, compute_type), kv_num_heads, d_k),
                                    group,
                                    d_k);
        auto v_4d =
            detail::repeat_heads_4d(detail::split_heads(detail::convert_to(value, compute_type), kv_num_heads, d_v),
                                    group,
                                    d_v);

        // GatedDeltaNet always applies the 1/sqrt(d_k) scale, so an explicit scale is folded into the query.
        if (scale != 0.0f) {
            const auto correction = v0::Constant::create(ov::element::f32,
                                                         ov::Shape{},
                                                         {scale * std::sqrt(static_cast<float>(d_k))});
            q_4d = std::make_shared<v1::Multiply>(q_4d, std::make_shared<v1::ConvertLike>(correction, q_4d));
        }

        ov::Output<ov::Node> gate;
        if (gating) {
            gate = detail::repeat_heads_3d(detail::convert_to(inputs[4], compute_type), group);
        } else {
            // The "delta" rule is "gated_delta" with a neutral decay (exp(0) == 1).
            const auto batch_seq = std::make_shared<v8::Gather>(query_shape,
                                                                v0::Constant::create(i64, ov::Shape{2}, {0, 1}),
                                                                axis0_1);
            const auto heads = v0::Constant::create(i64, ov::Shape{1}, {q_num_heads});
            const auto gate_shape = std::make_shared<v0::Concat>(ov::OutputVector{batch_seq, heads}, 0);
            const auto zero =
                std::make_shared<v1::ConvertLike>(v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f}), query);
            gate = std::make_shared<v3::Broadcast>(zero, gate_shape);
        }

        ov::Output<ov::Node> beta = detail::convert_to(inputs[5], compute_type);
        if (beta_last_dim == 1) {
            if (q_num_heads > 1) {
                const auto repeats =
                    v0::Constant::create(i64, ov::Shape{3}, std::vector<int64_t>{1, 1, q_num_heads});
                beta = std::make_shared<v0::Tile>(beta, repeats);
            }
        } else {
            beta = detail::repeat_heads_3d(beta, group);
        }

        ov::element::Type state_type = compute_type;
        ov::Output<ov::Node> state;
        if (has_past_state) {
            state_type = inputs[3].get_element_type();
            state = detail::repeat_state_heads(detail::convert_to(inputs[3], compute_type), group, d_k, d_v);
        } else {
            const auto tail = v0::Constant::create(i64, ov::Shape{3}, std::vector<int64_t>{q_num_heads, d_k, d_v});
            const auto state_shape = std::make_shared<v0::Concat>(ov::OutputVector{batch, tail}, 0);
            const auto zero =
                std::make_shared<v1::ConvertLike>(v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f}), query);
            state = std::make_shared<v3::Broadcast>(zero, state_shape);
        }

        const auto gdn = std::make_shared<ov::op::internal::GatedDeltaNet>(q_4d, k_4d, v_4d, state, gate, beta);

        const auto output_pattern = v0::Constant::create(i64, ov::Shape{3}, {0, 0, -1});
        const ov::Output<ov::Node> output = std::make_shared<v1::Reshape>(gdn->output(0), output_pattern, true);

        ov::Output<ov::Node> present_state = gdn->output(1);
        if (group > 1) {
            const auto pattern = v0::Constant::create(i64,
                                                      ov::Shape{5},
                                                      std::vector<int64_t>{0, kv_num_heads, group, d_k, d_v});
            const auto grouped = std::make_shared<v1::Reshape>(present_state, pattern, true);
            present_state = std::make_shared<v8::Gather>(grouped,
                                                         v0::Constant::create(i64, ov::Shape{}, {0}),
                                                         v0::Constant::create(i64, ov::Shape{1}, {2}));
        }
        present_state = detail::convert_to(present_state, state_type);

        if (node.get_outputs_size() < 2) {
            return {output};
        }
        return {output, present_state};
    }

    // ---- General path: a serializable OV Loop recurrence in f32. ----
    const auto f32 = ov::element::f32;
    const auto perm = v0::Constant::create(i64, ov::Shape{4}, std::vector<int64_t>{0, 2, 1, 3});

    // Per-q-head f32 tensors, GQA-expanded to H = q_num_heads: (B, T, H, .).
    const auto q4 = detail::split_heads(detail::convert_to(query, f32), q_num_heads, d_k);
    const auto k4 =
        detail::repeat_heads_4d(detail::split_heads(detail::convert_to(key, f32), kv_num_heads, d_k), group, d_k);
    const auto v4 =
        detail::repeat_heads_4d(detail::split_heads(detail::convert_to(value, f32), kv_num_heads, d_v), group, d_v);

    // gate expanded to per-key-dim (B, T, H, d_k).
    ov::Output<ov::Node> gate4;
    if (gating) {
        const auto decf = detail::convert_to(inputs[4], f32);
        if (decay_per_head) {
            const auto d4 =
                std::make_shared<v0::Unsqueeze>(decf, v0::Constant::create(i64, ov::Shape{1}, {3}));  // (B,T,kv,1)
            const auto tiled = std::make_shared<v0::Tile>(
                d4,
                v0::Constant::create(i64, ov::Shape{4}, std::vector<int64_t>{1, 1, 1, d_k}));  // (B,T,kv,d_k)
            gate4 = detail::repeat_heads_4d(tiled, group, d_k);
        } else {
            gate4 = detail::repeat_heads_4d(detail::split_heads(decf, kv_num_heads, d_k), group, d_k);
        }
    }

    // beta expanded to (B, T, H, 1).
    ov::Output<ov::Node> beta4;
    if (delta) {
        const auto bf = detail::convert_to(inputs[5], f32);
        ov::Output<ov::Node> bth;
        if (beta_last_dim == 1) {
            bth = std::make_shared<v0::Tile>(
                bf,
                v0::Constant::create(i64, ov::Shape{3}, std::vector<int64_t>{1, 1, q_num_heads}));  // (B,T,H)
        } else {
            bth = detail::repeat_heads_3d(bf, group);  // (B,T,H)
        }
        beta4 = std::make_shared<v0::Unsqueeze>(bth, v0::Constant::create(i64, ov::Shape{1}, {3}));  // (B,T,H,1)
    }

    // state (B, H, d_k, d_v) f32.
    ov::element::Type state_type = compute_type;
    ov::Output<ov::Node> state4;
    if (has_past_state) {
        state_type = inputs[3].get_element_type();
        state4 = detail::repeat_state_heads(detail::convert_to(inputs[3], f32), group, d_k, d_v);
    } else {
        const auto tail = v0::Constant::create(i64, ov::Shape{3}, std::vector<int64_t>{q_num_heads, d_k, d_v});
        const auto state_shape = std::make_shared<v0::Concat>(ov::OutputVector{batch, tail}, 0);
        state4 = std::make_shared<v3::Broadcast>(v0::Constant::create(f32, ov::Shape{}, {0.0f}), state_shape);
    }

    // Transpose (B, T, H, .) -> (B, H, T, .) and merge (B, H) -> (B*H).
    auto merge_seq = [&](const ov::Output<ov::Node>& x, int64_t last) -> ov::Output<ov::Node> {
        const auto t = std::make_shared<v1::Transpose>(x, perm);  // (B,H,S,last)
        const auto sh = std::make_shared<v3::ShapeOf>(t, i64);
        const auto b = std::make_shared<v8::Gather>(sh, v0::Constant::create(i64, ov::Shape{1}, {0}), axis0_1);
        const auto h = std::make_shared<v8::Gather>(sh, v0::Constant::create(i64, ov::Shape{1}, {1}), axis0_1);
        const auto s = std::make_shared<v8::Gather>(sh, v0::Constant::create(i64, ov::Shape{1}, {2}), axis0_1);
        const auto bh = std::make_shared<v1::Multiply>(b, h);
        const auto new_shape = std::make_shared<v0::Concat>(
            ov::OutputVector{bh, s, v0::Constant::create(i64, ov::Shape{1}, {last})},
            0);
        return std::make_shared<v1::Reshape>(t, new_shape, false);
    };
    auto merge_state = [&](const ov::Output<ov::Node>& x) -> ov::Output<ov::Node> {
        const auto sh = std::make_shared<v3::ShapeOf>(x, i64);
        const auto b = std::make_shared<v8::Gather>(sh, v0::Constant::create(i64, ov::Shape{1}, {0}), axis0_1);
        const auto h = std::make_shared<v8::Gather>(sh, v0::Constant::create(i64, ov::Shape{1}, {1}), axis0_1);
        const auto bh = std::make_shared<v1::Multiply>(b, h);
        const auto new_shape = std::make_shared<v0::Concat>(
            ov::OutputVector{bh,
                             v0::Constant::create(i64, ov::Shape{1}, {d_k}),
                             v0::Constant::create(i64, ov::Shape{1}, {d_v})},
            0);
        return std::make_shared<v1::Reshape>(x, new_shape, false);
    };

    const auto q_m = merge_seq(q4, d_k);
    const auto k_m = merge_seq(k4, d_k);
    const auto v_m = merge_seq(v4, d_v);
    const auto g_m = gating ? merge_seq(gate4, d_k) : ov::Output<ov::Node>{};
    const auto beta_m = delta ? merge_seq(beta4, 1) : ov::Output<ov::Node>{};
    const auto state_m = merge_state(state4);

    const float scale_value = (scale != 0.0f) ? scale : 1.0f / std::sqrt(static_cast<float>(d_k));
    const auto rec =
        detail::build_recurrence_loop(q_m, k_m, v_m, g_m, beta_m, state_m, scale_value, gating, delta);

    // output: (B*H, T, d_v) -> (B, H, T, d_v) -> (B, T, H, d_v) -> (B, T, H * d_v).
    const auto out_sh = std::make_shared<v3::ShapeOf>(rec.output, i64);
    const auto t_dim = std::make_shared<v8::Gather>(out_sh, v0::Constant::create(i64, ov::Shape{1}, {1}), axis0_1);
    const auto out_shape = std::make_shared<v0::Concat>(
        ov::OutputVector{batch,
                         v0::Constant::create(i64, ov::Shape{1}, {q_num_heads}),
                         t_dim,
                         v0::Constant::create(i64, ov::Shape{1}, {d_v})},
        0);
    const auto out_bhtd = std::make_shared<v1::Reshape>(rec.output, out_shape, false);  // (B,H,T,d_v)
    const auto out_bthd = std::make_shared<v1::Transpose>(out_bhtd, perm);              // (B,T,H,d_v)
    ov::Output<ov::Node> output =
        std::make_shared<v1::Reshape>(out_bthd, v0::Constant::create(i64, ov::Shape{3}, {0, 0, -1}), true);
    output = detail::convert_to(output, compute_type);

    // present_state: (B*H, d_k, d_v) -> (B, H, d_k, d_v), then reduce replicated heads to kv_num_heads.
    const auto ps_shape = std::make_shared<v0::Concat>(
        ov::OutputVector{batch,
                         v0::Constant::create(i64, ov::Shape{1}, {q_num_heads}),
                         v0::Constant::create(i64, ov::Shape{1}, {d_k}),
                         v0::Constant::create(i64, ov::Shape{1}, {d_v})},
        0);
    ov::Output<ov::Node> present_state = std::make_shared<v1::Reshape>(rec.present_state, ps_shape, false);
    if (group > 1) {
        const auto pattern =
            v0::Constant::create(i64, ov::Shape{5}, std::vector<int64_t>{0, kv_num_heads, group, d_k, d_v});
        const auto grouped = std::make_shared<v1::Reshape>(present_state, pattern, true);
        present_state = std::make_shared<v8::Gather>(grouped,
                                                     v0::Constant::create(i64, ov::Shape{}, {0}),
                                                     v0::Constant::create(i64, ov::Shape{1}, {2}));
    }
    present_state = detail::convert_to(present_state, state_type);

    if (node.get_outputs_size() < 2) {
        return {output};
    }
    return {output, present_state};
}

ONNX_OP("LinearAttention", OPSET_SINCE(27), ai_onnx::opset_27::linear_attention);
}  // namespace opset_27
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
