// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "builders/builder.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "builders/dequantize.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/runtime/properties.hpp"
#include "ov_ops/fully_connected_compressed.hpp"
#include "rt_info_keys.hpp"

namespace ov {
namespace frontend {
namespace gguf {

namespace {

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v4 = ov::op::v4;
namespace v6 = ov::op::v6;
namespace v8 = ov::op::v8;
namespace v13 = ov::op::v13;
namespace v15 = ov::op::v15;

using ov::Node;
using ov::Output;
using ov::OutputVector;
using ov::Shape;
using FullyConnectedCompressed = ov::op::internal::FullyConnectedCompressed;

constexpr int64_t INT64_MAX_SLICE = 9223372036854775807LL;

// Decoded qwen3 hyper-parameters (raw GGUF metadata, architecture-prefixed).
struct Qwen3Config {
    std::string arch;
    int hidden = 0;
    int head_num = 0;
    int head_num_kv = 0;
    int head_size = 0;
    int layer_num = 0;
    float rms_eps = 1e-6f;
    float rope_base = 10000.0f;
    int max_pos = 2048;
    int file_type = 1;
};

void set_name(const std::shared_ptr<ov::Node>& node, const std::string& name) {
    node->output(0).set_names({name});
    node->set_friendly_name(name);
}

std::string blk(int layer_idx, const std::string& suffix) {
    return "blk." + std::to_string(layer_idx) + "." + suffix;
}

// A canonical "absent optional input" placeholder, matching ConvertFCToCompressed output.
std::shared_ptr<ov::Node> empty_input() {
    return std::make_shared<v0::Constant>(ov::element::dynamic, ov::Shape{0});
}

// Build X * W^T keeping W compressed (raw gguf_* / f16 Constant) via FullyConnectedCompressed.
// Optional bias is fed through the op's bias input when present (qwen3 has none).
Output<Node> make_fc(const Output<Node>& input, const GGUFReader& reader, const std::string& base_name) {
    auto weight = reader.tensor_constant(base_name + ".weight");
    std::shared_ptr<ov::Node> bias = empty_input();
    if (reader.has_tensor(base_name + ".bias")) {
        bias = reader.tensor_constant(base_name + ".bias");
    }
    return std::make_shared<FullyConnectedCompressed>(input, weight, bias, empty_input(), empty_input());
}

// RMSNorm: x * rsqrt(mean(x^2, -1) + eps) * weight. `four_d` selects the rank used by the qwen3
// per-head q/k norm (operating on [B, S, H, D]) vs. the standard token-level norm ([B, S, H]).
Output<Node> make_rms_norm(const GGUFReader& reader,
                           const std::string& weight_base,
                           const Output<Node>& input,
                           float eps,
                           bool four_d) {
    auto eps_node = std::make_shared<v0::Constant>(ov::element::f32, ov::Shape{}, eps);
    auto exponent = std::make_shared<v0::Constant>(ov::element::f32, ov::Shape{}, 2.0f);
    auto square = std::make_shared<v1::Power>(input, exponent);
    auto reduce_axis = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{1}, -1);
    auto variance = std::make_shared<v1::ReduceMean>(square, reduce_axis, true);
    auto add_eps = std::make_shared<v1::Add>(variance, eps_node);
    auto sqrt_node = std::make_shared<v0::Sqrt>(add_eps);
    auto one = std::make_shared<v0::Constant>(ov::element::f32, ov::Shape{}, 1.0f);
    auto reciprocal = std::make_shared<v1::Divide>(one, sqrt_node);

    std::shared_ptr<ov::Node> mul =
        std::make_shared<v1::Multiply>(reciprocal, input, ov::op::AutoBroadcastType::NUMPY);

    if (reader.has_tensor(weight_base + ".weight")) {
        auto weight = reader.tensor_constant(weight_base + ".weight");
        const std::vector<int64_t> axes = four_d ? std::vector<int64_t>{0, 1, 2} : std::vector<int64_t>{0, 1};
        auto axes_const = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{axes.size()}, axes);
        auto weight_unsq = std::make_shared<v0::Unsqueeze>(weight, axes_const);
        auto weight_f32 = std::make_shared<v0::Convert>(weight_unsq, ov::element::f32);
        mul = std::make_shared<v1::Multiply>(mul, weight_f32, ov::op::AutoBroadcastType::NUMPY);
    }
    return mul;
}

// Reshape [B, S, hidden] -> [B, num_h, S, head_dim], optionally applying the qwen3 per-head RMSNorm
// (q_norm / k_norm) before the transpose when the corresponding weight tensor is present.
std::shared_ptr<v1::Transpose> split_heads(const Output<Node>& x,
                                           int num_h,
                                           int head_dim,
                                           const GGUFReader& reader,
                                           float eps,
                                           const std::string& norm_base) {
    auto shape = std::make_shared<v0::Constant>(ov::element::i64,
                                                ov::Shape{4},
                                                std::vector<int64_t>{0, 0, num_h, head_dim});
    auto reshaped = std::make_shared<v1::Reshape>(x, shape, true);
    Output<Node> normed = reshaped;
    if (reader.has_tensor(norm_base + ".weight")) {
        normed = make_rms_norm(reader, norm_base, reshaped, eps, /*four_d=*/true);
    }
    auto transpose_order =
        std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{0, 2, 1, 3});
    return std::make_shared<v1::Transpose>(normed, transpose_order);
}

// Rotate the second half of the head dimension and negate it (RoPE helper).
Output<Node> rotate_half(const Output<Node>& x, int64_t head_size, const Output<Node>& axis) {
    auto start_2nd = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{head_size / 2});
    auto stop_max = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{INT64_MAX_SLICE});
    auto step_1 = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto second = std::make_shared<v8::Slice>(x, start_2nd, stop_max, step_1, axis);
    auto neg_one = std::make_shared<v0::Constant>(ov::element::f32, ov::Shape{1, 1, 1, 1}, std::vector<float>{-1.0f});
    auto negated = std::make_shared<v1::Multiply>(second, neg_one);
    auto start_1st = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto stop_half = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{head_size / 2});
    auto step_1b = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto first = std::make_shared<v8::Slice>(x, start_1st, stop_half, step_1b, axis);
    return std::make_shared<v0::Concat>(OutputVector{negated, first}, -1);
}

std::tuple<Output<Node>, Output<Node>, std::pair<Output<Node>, Output<Node>>> apply_rotary_pos_emb(
    const Output<Node>& q,
    const Output<Node>& k,
    const Output<Node>& cos,
    const Output<Node>& sin,
    int64_t head_size,
    const Output<Node>& hidden_dim,
    const std::pair<Output<Node>, Output<Node>>& cos_sin_cached,
    int64_t unsqueeze_dim = 1) {
    Output<Node> cos_u, sin_u;
    if (cos_sin_cached.first.get_node() == nullptr) {
        auto axis_c = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{}, unsqueeze_dim);
        cos_u = std::make_shared<v0::Unsqueeze>(cos, axis_c);
        auto axis_s = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{}, unsqueeze_dim);
        sin_u = std::make_shared<v0::Unsqueeze>(sin, axis_s);
    } else {
        cos_u = cos_sin_cached.first;
        sin_u = cos_sin_cached.second;
    }
    auto q_rot = std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(q, cos_u),
                                           std::make_shared<v1::Multiply>(rotate_half(q, head_size, hidden_dim), sin_u));
    auto k_rot = std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(k, cos_u),
                                           std::make_shared<v1::Multiply>(rotate_half(k, head_size, hidden_dim), sin_u));
    return {q_rot, k_rot, {cos_u, sin_u}};
}

std::pair<Output<Node>, Output<Node>> rope_emb(const Output<Node>& rope_const,
                                               const Output<Node>& position_ids,
                                               const Output<Node>& batch_dim) {
    auto pos_axis = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{}, 1);
    auto position_expanded =
        std::make_shared<v0::Convert>(std::make_shared<v0::Unsqueeze>(position_ids, pos_axis), ov::element::f32);

    auto target_shape = std::make_shared<v0::Concat>(
        OutputVector{batch_dim,
                     std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, 1),
                     std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, 1)},
        0);
    auto inv_freq_expanded =
        std::make_shared<v3::Broadcast>(rope_const, target_shape, ov::op::BroadcastType::BIDIRECTIONAL);
    auto freqs = std::make_shared<v0::MatMul>(inv_freq_expanded, position_expanded, false, false);
    auto freqs_t = std::make_shared<v1::Transpose>(
        freqs,
        std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{3}, std::vector<int32_t>{0, 2, 1}));
    auto emb = std::make_shared<v0::Concat>(OutputVector{freqs_t, freqs_t}, -1);
    return {std::make_shared<v0::Cos>(emb), std::make_shared<v0::Sin>(emb)};
}

// Additive causal attention mask (with attention_mask integration). Ported verbatim from the
// reference graph; operates purely on shapes/indices, no weights involved.
Output<Node> causal_mask(const Output<Node>& attention_mask,
                         const Output<Node>& keys,
                         const Output<Node>& hidden_dim,
                         const Output<Node>& input_shape) {
    auto t130 = std::make_shared<v3::ShapeOf>(attention_mask, ov::element::i64);
    auto t131 = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{}, 1);
    auto t132 = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{}, 0);
    auto t133 = std::make_shared<v8::Gather>(t130, t131, t132);

    auto t134 = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto t135 = std::make_shared<v1::Reshape>(t133, t134, false);
    auto index_1 = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{}, 1);
    auto axis_0 = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{}, 0);
    auto t127 = std::make_shared<v8::Gather>(input_shape, index_1, axis_0);
    auto t129 = std::make_shared<v1::Reshape>(t127, t134, false);
    auto t136 = std::make_shared<v0::Concat>(OutputVector{t129, t135}, 0);
    auto min_shape_val = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 1});
    auto t137 = std::make_shared<v1::Maximum>(min_shape_val, t136, ov::op::AutoBroadcastType::NUMPY);
    auto const_neg = std::make_shared<v0::Constant>(ov::element::f32, ov::Shape{}, -65504.0f);
    auto t138 = std::make_shared<v3::Broadcast>(const_neg, t137, ov::op::BroadcastType::NUMPY);

    auto t139 = std::make_shared<v3::ShapeOf>(t138, ov::element::i32);
    auto t140 = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{}, 1);
    auto t141 = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{}, 0);
    auto t142 = std::make_shared<v8::Gather>(t139, t140, t141, 0);
    auto t143 = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{}, 1);

    auto zero_const = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{}, 0);
    auto t144 = std::make_shared<v4::Range>(zero_const, t142, t143, ov::element::i32);
    auto axes_zero = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, 0);
    auto t145 = std::make_shared<v0::Unsqueeze>(t144, axes_zero);
    auto t146 = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{}, 1);
    auto t147 = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{}, 0);
    auto t148 = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{}, 0);

    auto t149 = std::make_shared<v8::Gather>(t139, t147, t148);
    auto t150 = std::make_shared<v1::Add>(t149, t146, ov::op::AutoBroadcastType::NUMPY);
    auto t151 = std::make_shared<v4::Range>(t146, t150, t143, ov::element::i32);
    auto t152 = std::make_shared<v0::Unsqueeze>(t151, t143);
    auto t153 = std::make_shared<v1::GreaterEqual>(t145, t152, ov::op::AutoBroadcastType::NUMPY);

    auto t154 = std::make_shared<v0::Constant>(ov::element::f32, ov::Shape{}, 0.0f);
    auto t155 = std::make_shared<v1::Select>(t153, t138, t154, ov::op::AutoBroadcastType::NUMPY);

    auto t156 = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{}, 0);
    auto t157 = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{}, 1);
    auto t158 = std::make_shared<v4::Range>(t156, t133, t157, ov::element::f32);
    auto t159 = std::make_shared<v0::Convert>(t158, ov::element::i64);
    auto t160 = std::make_shared<v0::Convert>(t159, ov::element::f32);
    auto t161 = std::make_shared<v3::ShapeOf>(keys, ov::element::i64);
    auto t162 = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{}, 2);
    auto t163 = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{}, 0);
    auto t164 = std::make_shared<v8::Gather>(t161, t162, t163, 0);
    auto t165 = std::make_shared<v1::Add>(t164, t127, ov::op::AutoBroadcastType::NUMPY);
    auto t166 = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{}, 1);
    auto t167 = std::make_shared<v4::Range>(t164, t165, t166, ov::element::f32);
    auto t168 = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{-1, 1});
    auto t169 = std::make_shared<v1::Reshape>(t167, t168, false);
    auto t170 = std::make_shared<v1::Greater>(t160, t169, ov::op::AutoBroadcastType::NUMPY);
    auto t171 = std::make_shared<v0::Convert>(t170, ov::element::f32);

    auto t172 = std::make_shared<v1::Multiply>(t155, t171, ov::op::AutoBroadcastType::NUMPY);
    auto t173 = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{}, 0);
    auto t174 = std::make_shared<v0::Unsqueeze>(t172, t173);
    auto t48 = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{}, 1);
    auto t175 = std::make_shared<v0::Unsqueeze>(t174, t48);
    auto t41 = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto t42 = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{}, 0);
    auto t43 = std::make_shared<v8::Gather>(input_shape, t41, t42);
    auto t176 = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto t177 = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto t178 = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto t179 = std::make_shared<v0::Concat>(OutputVector{t43, t176, t177, t178}, 0);
    auto t180 = std::make_shared<v3::Broadcast>(t175, t179, ov::op::BroadcastType::BIDIRECTIONAL);
    auto t181 = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto t182 = std::make_shared<v1::Reshape>(t180, t181, false);
    auto t183 = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{}, 0);
    auto t184 = std::make_shared<v3::ShapeOf>(t180, ov::element::i64);
    auto t185 = std::make_shared<v1::ReduceProd>(t184, t183, false);
    auto t186 = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{}, 1);
    auto t187 = std::make_shared<v4::Range>(t183, t185, t186, ov::element::i64);
    auto t188 = std::make_shared<v1::Reshape>(t187, t184, false);
    auto t189 = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto t190 = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto t191 = std::make_shared<v8::Slice>(t188, t189, t135, t190, hidden_dim);
    auto t192 = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{-1, 1});
    auto t193 = std::make_shared<v1::Reshape>(t191, t192, false);
    auto t194 = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto t195 = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto t196 = std::make_shared<v8::Slice>(t180, t194, t135, t195, hidden_dim);

    auto t197 = std::make_shared<v0::Unsqueeze>(attention_mask, t48);
    auto t198 = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{}, 2);
    auto t199 = std::make_shared<v0::Unsqueeze>(t197, t198);
    auto t200 = std::make_shared<v0::Convert>(t199, ov::element::f32);
    auto t201 = std::make_shared<v1::Add>(t196, t200, ov::op::AutoBroadcastType::NUMPY);
    auto t202 = std::make_shared<v0::Constant>(ov::element::f32, ov::Shape{1, 1, 1, 1}, std::vector<float>{0.0f});
    auto t203 = std::make_shared<v1::Equal>(t201, t202, ov::op::AutoBroadcastType::NUMPY);
    auto t204 = std::make_shared<v0::Constant>(ov::element::f32, ov::Shape{}, -65504.0f);
    auto t205 = std::make_shared<v1::Select>(t203, t204, t196, ov::op::AutoBroadcastType::NUMPY);
    auto t206 = std::make_shared<v3::ShapeOf>(t196, ov::element::i64);
    auto t207 = std::make_shared<v3::Broadcast>(t205, t206, ov::op::BroadcastModeSpec(ov::op::BroadcastType::NUMPY));
    auto t208 = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto t209 = std::make_shared<v1::Reshape>(t207, t208, false);
    auto t210 = std::make_shared<v15::ScatterNDUpdate>(t182, t193, t209);
    auto t211 = std::make_shared<v1::Reshape>(t210, t184, false);
    auto t213 = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto t214 = std::make_shared<v1::Reshape>(t164, t213, false);
    auto t215 = std::make_shared<v1::Add>(t214, t129, ov::op::AutoBroadcastType::NUMPY);
    auto t212 = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto t216 = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto t217 = std::make_shared<v8::Slice>(t211, t212, t215, t216, hidden_dim);
    return t217;
}

// Attention block: split heads, RoPE, KV-cache (stateful ReadValue/Assign), optional GQA expansion,
// SDPA, and output reshape. Returns {context, {k_assign, v_assign}, cos_sin_cache, used_mask}.
std::tuple<Output<Node>, ov::SinkVector, std::pair<Output<Node>, Output<Node>>, Output<Node>>
multi_head_attention(const Output<Node>& query,
                     const Output<Node>& key,
                     const Output<Node>& value,
                     const GGUFReader& reader,
                     const Qwen3Config& cfg,
                     int layer_idx,
                     const Output<Node>& batch_dim,
                     const Output<Node>& hidden_dim,
                     const Output<Node>& input_shape,
                     const Output<Node>& output_shape,
                     const Output<Node>& attention_mask,
                     const Output<Node>& mask,
                     const Output<Node>& position_ids,
                     const Output<Node>& rope_const,
                     const Output<Node>& beam_idx,
                     const std::pair<Output<Node>, Output<Node>>& cos_sin_cached) {
    const int num_heads = cfg.head_num;
    const int head_dim = cfg.head_size;
    const int num_heads_kv = cfg.head_num_kv;

    auto q_split = split_heads(query, num_heads, head_dim, reader, cfg.rms_eps, blk(layer_idx, "attn_q_norm"));
    auto k_split = split_heads(key, num_heads_kv, head_dim, reader, cfg.rms_eps, blk(layer_idx, "attn_k_norm"));
    auto v_split = split_heads(value, num_heads_kv, head_dim, reader, cfg.rms_eps, blk(layer_idx, "attn_v_norm"));

    Output<Node> cos, sin;
    if (cos_sin_cached.first.get_node() == nullptr) {
        std::tie(cos, sin) = rope_emb(rope_const, position_ids, batch_dim);
    }
    auto [q_rot, k_rot, new_cos_sin] =
        apply_rotary_pos_emb(q_split, k_split, cos, sin, head_dim, hidden_dim, cos_sin_cached);

    auto create_cache = [&](const std::string& name, const Output<Node>& init_value) {
        auto var = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape{-1, num_heads_kv, -1, head_dim}, ov::element::f32, name});
        auto read_value = std::make_shared<v6::ReadValue>(init_value, var);
        auto gathered = std::make_shared<v8::Gather>(
            read_value, beam_idx, std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{}, 0), 0);
        return std::make_pair(var, gathered);
    };

    auto zero_const = std::make_shared<v0::Constant>(ov::element::f32, ov::Shape{}, 0.0f);
    auto kv_default_shape = [&]() {
        return std::make_shared<v0::Concat>(
            OutputVector{batch_dim,
                         std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, num_heads_kv),
                         std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, 0),
                         std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, head_dim)},
            0);
    };
    auto k_cache_default = std::make_shared<v3::Broadcast>(zero_const, kv_default_shape());
    auto v_cache_default = std::make_shared<v3::Broadcast>(zero_const, kv_default_shape());
    // The KV-cache default value is an empty (seq=0) tensor. If batch_dim resolves to a constant
    // (e.g. after a static reshape), constant folding would materialize a 0-element Constant and then
    // crash inside Concat::evaluate when this empty tensor is concatenated downstream. Keep the
    // ReadValue init subgraph dynamic so it is never folded.
    ov::pass::disable_constant_folding(k_cache_default);
    ov::pass::disable_constant_folding(v_cache_default);

    auto k_cache = create_cache("past_key_values." + std::to_string(layer_idx) + ".keypresent." +
                                    std::to_string(layer_idx) + ".key",
                                k_cache_default);
    auto v_cache = create_cache("past_key_values." + std::to_string(layer_idx) + ".valuepresent." +
                                    std::to_string(layer_idx) + ".key",
                                v_cache_default);

    auto k_combined = std::make_shared<v0::Concat>(OutputVector{k_cache.second, k_rot}, 2);
    auto v_combined = std::make_shared<v0::Concat>(OutputVector{v_cache.second, v_split}, 2);
    auto k_assign = std::make_shared<v6::Assign>(k_combined, k_cache.first);
    auto v_assign = std::make_shared<v6::Assign>(v_combined, v_cache.first);

    Output<Node> k_reshaped = k_combined;
    Output<Node> v_reshaped = v_combined;
    if (num_heads != num_heads_kv) {
        const int kv_per_head = num_heads / num_heads_kv;
        auto k_unsq =
            std::make_shared<v0::Unsqueeze>(k_combined, std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{}, 2));
        auto v_unsq =
            std::make_shared<v0::Unsqueeze>(v_combined, std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{}, 2));
        auto gqa_shape = [&]() {
            return std::make_shared<v0::Concat>(
                OutputVector{batch_dim,
                             std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, num_heads_kv),
                             std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, kv_per_head),
                             std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, 0),
                             std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, head_dim)},
                0);
        };
        auto reshape_target = [&]() {
            return std::make_shared<v0::Constant>(ov::element::i64,
                                                  ov::Shape{4},
                                                  std::vector<int64_t>{0, num_heads, -1, head_dim});
        };
        k_reshaped = std::make_shared<v1::Reshape>(
            std::make_shared<v3::Broadcast>(k_unsq, gqa_shape(), ov::op::BroadcastType::BIDIRECTIONAL),
            reshape_target(),
            true);
        v_reshaped = std::make_shared<v1::Reshape>(
            std::make_shared<v3::Broadcast>(v_unsq, gqa_shape(), ov::op::BroadcastType::BIDIRECTIONAL),
            reshape_target(),
            true);
    }

    Output<Node> final_mask = mask;
    if (mask.get_node() == nullptr) {
        final_mask = causal_mask(attention_mask, k_cache.second, hidden_dim, input_shape);
    }

    auto attention =
        std::make_shared<v13::ScaledDotProductAttention>(q_rot, k_reshaped, v_reshaped, final_mask, false);
    auto transpose_order =
        std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{0, 2, 1, 3});
    auto context_transposed = std::make_shared<v1::Transpose>(attention, transpose_order);
    auto output = std::make_shared<v1::Reshape>(context_transposed, output_shape, false);
    return {output, ov::SinkVector{k_assign, v_assign}, new_cos_sin, final_mask};
}

// One transformer block: pre-attn norm -> attention -> residual -> post-attn norm -> SwiGLU MLP ->
// residual.
std::tuple<Output<Node>, ov::SinkVector, Output<Node>, std::pair<Output<Node>, Output<Node>>, std::shared_ptr<ov::Node>>
build_layer(const GGUFReader& reader,
            const Qwen3Config& cfg,
            int layer_idx,
            const Output<Node>& hidden_states,
            const Output<Node>& attn_mask,
            const Output<Node>& causal_mask_in,
            const Output<Node>& position_ids,
            const Output<Node>& rope_const,
            const Output<Node>& beam_idx,
            const Output<Node>& batch_dim,
            const Output<Node>& hidden_dim,
            const std::pair<Output<Node>, Output<Node>>& cos_sin_cached,
            const std::shared_ptr<ov::Node>& output_shape) {
    auto input_layernorm = make_rms_norm(reader, blk(layer_idx, "attn_norm"), hidden_states, cfg.rms_eps, false);

    auto q = make_fc(input_layernorm, reader, blk(layer_idx, "attn_q"));
    auto k = make_fc(input_layernorm, reader, blk(layer_idx, "attn_k"));
    auto v = make_fc(input_layernorm, reader, blk(layer_idx, "attn_v"));

    std::shared_ptr<ov::Node> final_output_shape = output_shape;
    if (!output_shape) {
        auto shape_of = std::make_shared<v3::ShapeOf>(input_layernorm);
        auto indices = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{0, 1});
        auto gathered = std::make_shared<v8::Gather>(
            shape_of, indices, std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{}, 0));
        auto minus_one = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, -1);
        final_output_shape = std::make_shared<v0::Concat>(OutputVector{gathered, minus_one}, 0);
    }

    auto input_shape_node = std::make_shared<v3::ShapeOf>(input_layernorm);
    auto [attn_output, sinks, new_cos_sin, new_mask] = multi_head_attention(q,
                                                                            k,
                                                                            v,
                                                                            reader,
                                                                            cfg,
                                                                            layer_idx,
                                                                            batch_dim,
                                                                            hidden_dim,
                                                                            input_shape_node,
                                                                            final_output_shape,
                                                                            attn_mask,
                                                                            causal_mask_in,
                                                                            position_ids,
                                                                            rope_const,
                                                                            beam_idx,
                                                                            cos_sin_cached);

    auto o_proj = make_fc(attn_output, reader, blk(layer_idx, "attn_output"));
    auto attn_add = std::make_shared<v1::Add>(hidden_states, o_proj, ov::op::AutoBroadcastType::NUMPY);

    auto post_attn_norm = make_rms_norm(reader, blk(layer_idx, "ffn_norm"), attn_add, cfg.rms_eps, false);

    auto gate_proj = make_fc(post_attn_norm, reader, blk(layer_idx, "ffn_gate"));
    auto silu = std::make_shared<v4::Swish>(gate_proj);
    auto up_proj = make_fc(post_attn_norm, reader, blk(layer_idx, "ffn_up"));
    auto mlp_mul = std::make_shared<v1::Multiply>(silu, up_proj, ov::op::AutoBroadcastType::NUMPY);
    auto down_proj = make_fc(mlp_mul, reader, blk(layer_idx, "ffn_down"));

    auto output = std::make_shared<v1::Add>(attn_add, down_proj, ov::op::AutoBroadcastType::NUMPY);
    return {output, sinks, new_mask, new_cos_sin, final_output_shape};
}

// inv_freq[i] = 1 / base^(2i/head_dim), shape [1, head_dim/2, 1].
std::shared_ptr<ov::Node> init_rope(int head_dim, float base) {
    const size_t num = static_cast<size_t>(head_dim) / 2;
    std::vector<float> inv_freq(num);
    for (size_t i = 0; i < num; ++i) {
        inv_freq[i] = 1.0f / std::pow(base, static_cast<float>(2 * i) / static_cast<float>(head_dim));
    }
    return std::make_shared<v0::Constant>(ov::element::f32, ov::Shape{1, num, 1}, inv_freq);
}

std::shared_ptr<ov::Model> build_language_model(const GGUFReader& reader, const Qwen3Config& cfg) {
    auto input_ids = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
    set_name(input_ids, "input_ids");
    auto attention_mask = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
    set_name(attention_mask, "attention_mask");
    auto position_ids = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
    set_name(position_ids, "position_ids");
    auto beam_idx = std::make_shared<v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    set_name(beam_idx, "beam_idx");

    // Embedding (Option A): the embedding tensor is the only weight materialized to dense f16 at
    // load time; Gather it, then convert to the f32 activation precision used by the rest of the graph.
    auto embed_f16 = dequantize_to_f16(reader, "token_embd.weight");
    auto ids_i32 = std::make_shared<v0::Convert>(input_ids, ov::element::i32);
    auto gather_axis = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{}, 0);
    auto gathered = std::make_shared<v8::Gather>(embed_f16, ids_i32, gather_axis);
    Output<Node> hidden_states = std::make_shared<v0::Convert>(gathered, ov::element::f32);

    auto rope_const = init_rope(cfg.head_size, cfg.rope_base);

    auto input_shape = std::make_shared<v3::ShapeOf>(input_ids);
    auto batch_axis = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, 0);
    auto batch_size = std::make_shared<v8::Gather>(input_shape, batch_axis, batch_axis);
    auto hidden_dim = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, 3);

    ov::SinkVector sinks;
    Output<Node> causal_mask_state;
    std::pair<Output<Node>, Output<Node>> cos_sin_cached;
    std::shared_ptr<ov::Node> output_shape = nullptr;

    for (int i = 0; i < cfg.layer_num; ++i) {
        auto [new_hidden, layer_sinks, new_mask, new_cos_sin, new_shape] = build_layer(reader,
                                                                                       cfg,
                                                                                       i,
                                                                                       hidden_states,
                                                                                       attention_mask,
                                                                                       causal_mask_state,
                                                                                       position_ids,
                                                                                       rope_const,
                                                                                       beam_idx,
                                                                                       batch_size,
                                                                                       hidden_dim,
                                                                                       cos_sin_cached,
                                                                                       output_shape);
        hidden_states = new_hidden;
        causal_mask_state = new_mask;
        cos_sin_cached = new_cos_sin;
        output_shape = new_shape;
        sinks.insert(sinks.end(), layer_sinks.begin(), layer_sinks.end());
    }

    auto final_norm = make_rms_norm(reader, "output_norm", hidden_states, cfg.rms_eps, false);

    // lm_head: dedicated output projection when present, otherwise tied to the (compressed) embedding
    // weight. Both stay compressed via FullyConnectedCompressed (no embedding dequant on this path).
    const std::string lm_head_base = reader.has_tensor("output.weight") ? "output" : "token_embd";
    auto logits_node = make_fc(final_norm, reader, lm_head_base);

    auto logits = std::make_shared<v0::Result>(logits_node);
    set_name(logits, "logits");

    ov::ParameterVector inputs{input_ids, attention_mask, position_ids, beam_idx};
    auto model = std::make_shared<ov::Model>(ov::OutputVector{logits->output(0)}, sinks, inputs);

    if (cfg.file_type == 0 || cfg.file_type == 1) {
        model->set_rt_info(ov::element::f16, {"runtime_options", ov::hint::kv_cache_precision.name()});
    }
    model->set_rt_info(8.0f, {"runtime_options", ov::hint::activations_scale_factor.name()});
    return model;
}

Qwen3Config read_config(const GGUFReader& reader) {
    Qwen3Config cfg;
    cfg.arch = reader.architecture();
    const auto key = [&](const char* suffix) {
        return arch_key(cfg.arch, suffix);
    };
    cfg.hidden = static_cast<int>(reader.get_u64(key(arch_suffix::embedding_length)));
    cfg.head_num = static_cast<int>(reader.get_u64(key(arch_suffix::attention_head_count)));
    cfg.head_num_kv = static_cast<int>(reader.get_u64(key(arch_suffix::attention_head_count_kv)));
    cfg.head_size = reader.has(key(arch_suffix::attention_key_length))
                        ? static_cast<int>(reader.get_u64(key(arch_suffix::attention_key_length)))
                        : cfg.hidden / cfg.head_num;
    cfg.layer_num = static_cast<int>(reader.get_u64(key(arch_suffix::block_count)));
    cfg.rms_eps = static_cast<float>(reader.get_f64(key(arch_suffix::attention_layer_norm_rms_eps)));
    cfg.rope_base = static_cast<float>(reader.get_f64(key(arch_suffix::rope_freq_base), 10000.0));
    cfg.max_pos = static_cast<int>(reader.get_u64(key(arch_suffix::context_length), 2048));
    cfg.file_type = static_cast<int>(reader.get_u64(file_keys::general_file_type, 1));

    OPENVINO_ASSERT(cfg.head_num > 0 && cfg.head_num_kv > 0 && cfg.head_size > 0 && cfg.layer_num > 0,
                    "[GGUF Frontend] Invalid qwen3 configuration decoded from metadata (head_num=",
                    cfg.head_num,
                    ", head_num_kv=",
                    cfg.head_num_kv,
                    ", head_size=",
                    cfg.head_size,
                    ", layer_num=",
                    cfg.layer_num,
                    ").");
    OPENVINO_ASSERT(cfg.head_num % cfg.head_num_kv == 0,
                    "[GGUF Frontend] head_num (",
                    cfg.head_num,
                    ") must be a multiple of head_num_kv (",
                    cfg.head_num_kv,
                    ").");
    return cfg;
}

}  // namespace

std::shared_ptr<ov::Model> build_qwen3_model(const GGUFReader& reader) {
    const Qwen3Config cfg = read_config(reader);
    return build_language_model(reader, cfg);
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
