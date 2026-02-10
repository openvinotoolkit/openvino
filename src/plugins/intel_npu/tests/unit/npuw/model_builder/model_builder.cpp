// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "model_builder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

#include "openvino/op/assign.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/sink.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset11.hpp"

namespace ov {
namespace test {
namespace npuw {

// ============================================================================
// Weight Functor Implementations
// ============================================================================

ov::Output<ov::Node> FP32Weight::operator()(const std::string& name,
                                            size_t rows,
                                            size_t cols,
                                            ov::element::Type compute_precision) const {
    auto weight =
        ov::opset11::Constant::create(compute_precision, ov::Shape{rows, cols}, std::vector<float>(rows * cols, 0.01f));
    weight->set_friendly_name(name);
    return weight->output(0);
}

ov::Output<ov::Node> FP16Weight::operator()(const std::string& name,
                                            size_t rows,
                                            size_t cols,
                                            ov::element::Type compute_precision) const {
    auto weight =
        ov::opset11::Constant::create(ov::element::f16, ov::Shape{rows, cols}, std::vector<float>(rows * cols, 0.01f));
    weight->set_friendly_name(name);

    auto convert = std::make_shared<ov::opset11::Convert>(weight, compute_precision);
    convert->set_friendly_name(name + "_convert");
    return convert->output(0);
}

ov::Output<ov::Node> CompressedWeight::operator()(const std::string& name,
                                                  size_t rows,
                                                  size_t cols,
                                                  ov::element::Type compute_precision) const {
    auto weight =
        ov::opset11::Constant::create(storage_type, ov::Shape{rows, cols}, std::vector<int8_t>(rows * cols, 1));
    weight->set_friendly_name(name);

    auto convert = std::make_shared<ov::opset11::Convert>(weight, ov::element::f16);
    convert->set_friendly_name(name + "_convert");

    auto scale = ov::opset11::Constant::create(ov::element::f16, ov::Shape{rows, 1}, std::vector<float>(rows, 0.01f));
    scale->set_friendly_name(name + "_scale");

    auto scaled = std::make_shared<ov::opset11::Multiply>(convert, scale);
    scaled->set_friendly_name(name + "_decompress");

    if (compute_precision != ov::element::f16) {
        auto to_compute = std::make_shared<ov::opset11::Convert>(scaled, compute_precision);
        to_compute->set_friendly_name(name + "_to_compute");
        return to_compute->output(0);
    }
    return scaled->output(0);
}

// ============================================================================
// Norm Functor Implementations
// ============================================================================

ov::Output<ov::Node> LayerNorm::operator()(const ov::Output<ov::Node>& input, const std::string& name) const {
    auto weight =
        ov::opset11::Constant::create(precision, ov::Shape{hidden_size}, std::vector<float>(hidden_size, 1.0f));
    weight->set_friendly_name(name + ".weight");

    auto bias = ov::opset11::Constant::create(precision, ov::Shape{hidden_size}, std::vector<float>(hidden_size, 0.0f));
    bias->set_friendly_name(name + ".bias");

    auto axes = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {-1});

    auto mvn = std::make_shared<ov::opset11::MVN>(input, axes, true, eps, ov::op::MVNEpsMode::INSIDE_SQRT);
    mvn->set_friendly_name(name + "_mvn");

    auto mul = std::make_shared<ov::opset11::Multiply>(mvn, weight);

    auto add = std::make_shared<ov::opset11::Add>(mul, bias);
    add->set_friendly_name(name);

    return add->output(0);
}

ov::Output<ov::Node> RMSNorm::operator()(const ov::Output<ov::Node>& input, const std::string& name) const {
    auto weight =
        ov::opset11::Constant::create(precision, ov::Shape{hidden_size}, std::vector<float>(hidden_size, 1.0f));
    weight->set_friendly_name(name + ".weight");

    auto squared = std::make_shared<ov::opset11::Multiply>(input, input);

    auto axes = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {-1});

    auto mean = std::make_shared<ov::opset11::ReduceMean>(squared, axes, true);

    auto eps_const = ov::opset11::Constant::create(precision, ov::Shape{}, {eps});

    auto mean_eps = std::make_shared<ov::opset11::Add>(mean, eps_const);

    auto rsqrt = std::make_shared<ov::opset11::Sqrt>(mean_eps);

    auto normalized = std::make_shared<ov::opset11::Divide>(input, rsqrt);

    auto scaled = std::make_shared<ov::opset11::Multiply>(normalized, weight);
    scaled->set_friendly_name(name);

    return scaled->output(0);
}

// ============================================================================
// RoPE Functor Implementations
// ============================================================================

RoPEEmbeddings gather_rope_embeddings(size_t head_dim,
                                      size_t max_position,
                                      ov::element::Type precision,
                                      const ov::Output<ov::Node>& position_ids,
                                      const std::string& name) {
    auto cos_table = ov::opset11::Constant::create(precision,
                                                   ov::Shape{max_position, head_dim},
                                                   std::vector<float>(max_position * head_dim, 0.5f));
    cos_table->set_friendly_name(name + ".cos_table");

    auto sin_table = ov::opset11::Constant::create(precision,
                                                   ov::Shape{max_position, head_dim},
                                                   std::vector<float>(max_position * head_dim, 0.5f));
    sin_table->set_friendly_name(name + ".sin_table");

    auto gather_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});

    auto cos_embed = std::make_shared<ov::opset11::Gather>(cos_table, position_ids, gather_axis, 0);
    cos_embed->set_friendly_name(name + "_cos_gather");

    auto sin_embed = std::make_shared<ov::opset11::Gather>(sin_table, position_ids, gather_axis, 0);
    sin_embed->set_friendly_name(name + "_sin_gather");

    auto unsqueeze_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {2});

    auto cos_unsqueezed = std::make_shared<ov::opset11::Unsqueeze>(cos_embed, unsqueeze_axis);
    cos_unsqueezed->set_friendly_name(name + "_cos_unsqueeze");

    auto sin_unsqueezed = std::make_shared<ov::opset11::Unsqueeze>(sin_embed, unsqueeze_axis);
    sin_unsqueezed->set_friendly_name(name + "_sin_unsqueeze");

    return {cos_unsqueezed->output(0), sin_unsqueezed->output(0)};
}

ov::Output<ov::Node> HalfRotationRoPE::operator()(const ov::Output<ov::Node>& input,
                                                  const ov::Output<ov::Node>& position_ids,
                                                  const std::string& name) const {
    auto [cos, sin] = gather_rope_embeddings(head_dim, max_position, precision, position_ids, name);

    const int64_t half = static_cast<int64_t>(head_dim / 2);
    auto zero = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto half_const = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {half});
    auto head_dim_const =
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(head_dim)});
    auto last_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    auto step = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1});

    auto x1 = std::make_shared<ov::op::v8::Slice>(input, zero, half_const, step, last_axis);
    x1->set_friendly_name(name + "_x1");

    auto x2 = std::make_shared<ov::op::v8::Slice>(input, half_const, head_dim_const, step, last_axis);
    x2->set_friendly_name(name + "_x2");

    auto neg_one = ov::opset11::Constant::create(precision, ov::Shape{}, {-1.0f});

    auto neg_x2 = std::make_shared<ov::opset11::Multiply>(x2, neg_one);
    neg_x2->set_friendly_name(name + "_neg_x2");

    auto rotated = std::make_shared<ov::opset11::Concat>(ov::OutputVector{neg_x2, x1}, -1);
    rotated->set_friendly_name(name + "_rotated");

    auto input_cos = std::make_shared<ov::opset11::Multiply>(input, cos);
    input_cos->set_friendly_name(name + "_input_cos");

    auto rotated_sin = std::make_shared<ov::opset11::Multiply>(rotated, sin);
    rotated_sin->set_friendly_name(name + "_rotated_sin");

    auto output = std::make_shared<ov::opset11::Add>(input_cos, rotated_sin);
    output->set_friendly_name(name);

    return output->output(0);
}

ov::Output<ov::Node> InterleavedRoPE::operator()(const ov::Output<ov::Node>& input,
                                                 const ov::Output<ov::Node>& position_ids,
                                                 const std::string& name) const {
    auto [cos, sin] = gather_rope_embeddings(head_dim, max_position, precision, position_ids, name);

    const int64_t half_dim = static_cast<int64_t>(head_dim / 2);
    auto reshape_5d =
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 0, 0, half_dim, 2});

    auto reshaped = std::make_shared<ov::opset11::Reshape>(input, reshape_5d, true);
    reshaped->set_friendly_name(name + "_reshape_5d");

    auto zero = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto one = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto two = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    auto step = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto last_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {-1});

    auto x_even = std::make_shared<ov::op::v8::Slice>(reshaped, zero, one, step, last_axis);
    x_even->set_friendly_name(name + "_x_even");

    auto x_odd = std::make_shared<ov::op::v8::Slice>(reshaped, one, two, step, last_axis);
    x_odd->set_friendly_name(name + "_x_odd");

    auto neg_one = ov::opset11::Constant::create(precision, ov::Shape{}, {-1.0f});

    auto neg_x_odd = std::make_shared<ov::opset11::Multiply>(x_odd, neg_one);
    neg_x_odd->set_friendly_name(name + "_neg_x_odd");

    auto rotated_pairs = std::make_shared<ov::opset11::Concat>(ov::OutputVector{neg_x_odd, x_even}, -1);
    rotated_pairs->set_friendly_name(name + "_rotated_pairs");

    auto reshape_4d = ov::opset11::Constant::create(ov::element::i64,
                                                    ov::Shape{4},
                                                    std::vector<int64_t>{0, 0, 0, static_cast<int64_t>(head_dim)});

    auto rotated = std::make_shared<ov::opset11::Reshape>(rotated_pairs, reshape_4d, true);
    rotated->set_friendly_name(name + "_rotated");

    auto input_cos = std::make_shared<ov::opset11::Multiply>(input, cos);
    input_cos->set_friendly_name(name + "_input_cos");

    auto rotated_sin = std::make_shared<ov::opset11::Multiply>(rotated, sin);
    rotated_sin->set_friendly_name(name + "_rotated_sin");

    auto output = std::make_shared<ov::opset11::Add>(input_cos, rotated_sin);
    output->set_friendly_name(name);

    return output->output(0);
}

// ============================================================================
// PositionIds Functor Implementations
// ============================================================================

ov::Output<ov::Node> Input2DPositionIds::operator()() const {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
    param->set_friendly_name("position_ids");
    param->output(0).set_names({"position_ids"});
    return param->output(0);
}

ov::Output<ov::Node> Input3DPositionIds::operator()() const {
    // Create [3, batch, seq] parameter (Qwen2.5-VL m-rope format)
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{3, -1, -1});
    param->set_friendly_name("position_ids");
    param->output(0).set_names({"position_ids"});

    // Extract section 0 -> [1, batch, seq] then squeeze to [batch, seq]
    auto indices = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto gather = std::make_shared<ov::opset11::Gather>(param, indices, axis);

    auto squeeze_axes = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto squeeze = std::make_shared<ov::opset11::Squeeze>(gather, squeeze_axes);
    squeeze->set_friendly_name("position_ids_2d");

    return squeeze->output(0);
}

// ============================================================================
// Free Building-Block Functions
// ============================================================================

ov::Output<ov::Node> make_linear(const ov::Output<ov::Node>& input,
                                 size_t in_features,
                                 size_t out_features,
                                 const std::string& name,
                                 ov::element::Type precision,
                                 bool add_bias,
                                 const WeightFn& weight_fn) {
    auto weight_output = weight_fn(name + ".weight", out_features, in_features, precision);

    auto matmul = std::make_shared<ov::opset11::MatMul>(input, weight_output, false, true);
    matmul->set_friendly_name(name);

    if (add_bias) {
        auto bias =
            ov::opset11::Constant::create(precision, ov::Shape{out_features}, std::vector<float>(out_features, 0.0f));
        bias->set_friendly_name(name + ".bias");

        auto add = std::make_shared<ov::opset11::Add>(matmul, bias);
        add->set_friendly_name(name + "_bias_add");
        return add->output(0);
    }

    return matmul->output(0);
}

ov::Output<ov::Node> make_multihead_reshape(const ov::Output<ov::Node>& input,
                                            size_t num_heads,
                                            size_t head_dim,
                                            const std::string& name) {
    auto shape = ov::opset11::Constant::create(
        ov::element::i64,
        ov::Shape{4},
        std::vector<int64_t>{0, -1, static_cast<int64_t>(num_heads), static_cast<int64_t>(head_dim)});

    auto reshape = std::make_shared<ov::opset11::Reshape>(input, shape, true);
    reshape->set_friendly_name(name);

    return reshape->output(0);
}

ov::Output<ov::Node> make_attention_transpose(const ov::Output<ov::Node>& input, const std::string& name) {
    auto order_const = ov::opset11::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});

    auto transpose = std::make_shared<ov::opset11::Transpose>(input, order_const);
    transpose->set_friendly_name(name);

    return transpose->output(0);
}

ov::Output<ov::Node> make_repeat_kv(const ov::Output<ov::Node>& kv,
                                    size_t num_heads,
                                    size_t num_kv_heads,
                                    size_t head_dim,
                                    const std::string& name) {
    if (num_kv_heads == 0 || num_heads == num_kv_heads) {
        return kv;
    }

    const size_t n_rep = num_heads / num_kv_heads;

    auto shape_of_kv = std::make_shared<ov::opset11::ShapeOf>(kv, ov::element::i64);

    auto gather_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});

    auto idx_01 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
    auto batch_kv_heads = std::make_shared<ov::opset11::Gather>(shape_of_kv, idx_01, gather_axis);

    auto n_rep_const = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(n_rep)});

    auto idx_23 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{2}, {2, 3});
    auto seq_head_dim = std::make_shared<ov::opset11::Gather>(shape_of_kv, idx_23, gather_axis);

    auto broadcast_shape =
        std::make_shared<ov::opset11::Concat>(ov::OutputVector{batch_kv_heads, n_rep_const, seq_head_dim}, 0);
    broadcast_shape->set_friendly_name(name + "_broadcast_shape");

    auto unsqueeze_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {2});

    auto unsqueezed = std::make_shared<ov::opset11::Unsqueeze>(kv, unsqueeze_axis);
    unsqueezed->set_friendly_name(name + "_unsqueeze");

    auto broadcasted =
        std::make_shared<ov::op::v3::Broadcast>(unsqueezed, broadcast_shape, ov::op::BroadcastType::BIDIRECTIONAL);
    broadcasted->set_friendly_name(name + "_broadcast");

    auto new_shape = ov::opset11::Constant::create(
        ov::element::i64,
        ov::Shape{4},
        std::vector<int64_t>{0, static_cast<int64_t>(num_heads), -1, static_cast<int64_t>(head_dim)});
    new_shape->set_friendly_name(name + "_shape");

    auto reshaped = std::make_shared<ov::opset11::Reshape>(broadcasted, new_shape, true);
    reshaped->set_friendly_name(name);

    return reshaped->output(0);
}

KVCacheResult make_kv_cache_concat(const ov::Output<ov::Node>& current_kv,
                                   const ov::Output<ov::Node>& batch_source,
                                   const ov::Output<ov::Node>& beam_idx,
                                   size_t num_heads,
                                   size_t head_dim,
                                   const std::string& name,
                                   ov::element::Type precision) {
    auto var_shape = ov::PartialShape{-1, static_cast<int64_t>(num_heads), -1, static_cast<int64_t>(head_dim)};
    auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{var_shape, precision, name});

    auto shape_of = std::make_shared<ov::opset11::ShapeOf>(batch_source, ov::element::i64);
    shape_of->set_friendly_name(name + "_shapeof");

    auto zero_idx = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto gather_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});

    auto batch_dim = std::make_shared<ov::opset11::Gather>(shape_of, zero_idx, gather_axis);
    batch_dim->set_friendly_name(name + "_batch_dim");

    auto num_heads_const = ov::opset11::Constant::create(ov::element::i64,
                                                         ov::Shape{1},
                                                         std::vector<int64_t>{static_cast<int64_t>(num_heads)});
    auto zero_seq = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto head_dim_const = ov::opset11::Constant::create(ov::element::i64,
                                                        ov::Shape{1},
                                                        std::vector<int64_t>{static_cast<int64_t>(head_dim)});

    auto init_shape = std::make_shared<ov::opset11::Concat>(ov::OutputVector{batch_dim->output(0),
                                                                             num_heads_const->output(0),
                                                                             zero_seq->output(0),
                                                                             head_dim_const->output(0)},
                                                            0);
    init_shape->set_friendly_name(name + "_init_shape");

    auto zero_scalar = ov::opset11::Constant::create(precision, ov::Shape{}, std::vector<float>{0.0f});

    auto init_value = std::make_shared<ov::opset11::Broadcast>(zero_scalar, init_shape);
    init_value->set_friendly_name(name + "_init");

    auto read_value = std::make_shared<ov::op::v6::ReadValue>(init_value, variable);
    read_value->set_friendly_name(name + "_read");

    auto beam_gather_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});

    auto beam_gather = std::make_shared<ov::opset11::Gather>(read_value, beam_idx, beam_gather_axis);
    beam_gather->set_friendly_name(name + "_beam_gather");

    auto concat = std::make_shared<ov::opset11::Concat>(ov::OutputVector{beam_gather->output(0), current_kv}, 2);
    concat->set_friendly_name(name + "_concat");

    auto assign = std::make_shared<ov::op::v6::Assign>(concat, variable);
    assign->set_friendly_name(name + "_assign");

    KVCacheResult result;
    result.concatenated = concat->output(0);
    result.assign = assign;
    return result;
}

ov::Output<ov::Node> make_sdpa(const ov::Output<ov::Node>& q,
                               const ov::Output<ov::Node>& k,
                               const ov::Output<ov::Node>& v,
                               const std::string& name,
                               const ov::Output<ov::Node>& attention_mask) {
    std::shared_ptr<ov::op::v13::ScaledDotProductAttention> sdpa;
    if (attention_mask.get_node()) {
        sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, attention_mask, false);
    } else {
        sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, false);
    }
    sdpa->set_friendly_name(name + ".sdpa");

    return sdpa->output(0);
}

ov::Output<ov::Node> make_embedding(const ov::Output<ov::Node>& input_ids,
                                    size_t vocab_size,
                                    size_t hidden_size,
                                    const std::string& name,
                                    ov::element::Type precision) {
    auto weight = ov::opset11::Constant::create(precision,
                                                ov::Shape{vocab_size, hidden_size},
                                                std::vector<float>(vocab_size * hidden_size, 0.01f));
    weight->set_friendly_name(name + ".weight");

    auto axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});

    auto gather = std::make_shared<ov::opset11::Gather>(weight, input_ids, axis, 0);
    gather->set_friendly_name(name);

    return gather->output(0);
}

ov::Output<ov::Node> make_lm_head(const ov::Output<ov::Node>& hidden_states,
                                  size_t hidden_size,
                                  size_t vocab_size,
                                  const std::string& name,
                                  ov::element::Type precision,
                                  const WeightFn& weight_fn) {
    return make_linear(hidden_states, hidden_size, vocab_size, name, precision, false, weight_fn);
}

// ============================================================================
// FFN Functor Implementations
// ============================================================================

ov::Output<ov::Node> SwiGLU::operator()(const ov::Output<ov::Node>& input, const std::string& name) const {
    auto gate = make_linear(input, hidden_size, intermediate_size, name + ".gate_proj", precision, false, weight_fn);
    auto up = make_linear(input, hidden_size, intermediate_size, name + ".up_proj", precision, false, weight_fn);

    auto sigmoid = std::make_shared<ov::opset11::Sigmoid>(gate);

    auto silu = std::make_shared<ov::opset11::Multiply>(gate, sigmoid);
    silu->set_friendly_name(name + "_silu");

    auto gate_up = std::make_shared<ov::opset11::Multiply>(silu, up);
    gate_up->set_friendly_name(name + "_gate_up");

    auto down = make_linear(gate_up, intermediate_size, hidden_size, name + ".down_proj", precision, false, weight_fn);

    return down;
}

ov::Output<ov::Node> GELUFn::operator()(const ov::Output<ov::Node>& input, const std::string& name) const {
    auto up = make_linear(input, hidden_size, intermediate_size, name + ".up_proj", precision, false, weight_fn);

    auto gelu = std::make_shared<ov::opset11::Gelu>(up);
    gelu->set_friendly_name(name + "_gelu");

    auto down = make_linear(gelu, intermediate_size, hidden_size, name + ".down_proj", precision, false, weight_fn);

    return down;
}

// ============================================================================
// SDPAttention Functor Implementation
// ============================================================================

LayerResult SDPAttention::operator()(const ov::Output<ov::Node>& input, const std::string& prefix) const {
    const size_t kv_heads = num_kv_heads;

    // Q, K, V projections
    auto q =
        make_linear(input, hidden_size, num_heads * head_dim, prefix + "self_attn.q_proj", precision, false, weight_fn);
    auto k =
        make_linear(input, hidden_size, kv_heads * head_dim, prefix + "self_attn.k_proj", precision, false, weight_fn);
    auto v =
        make_linear(input, hidden_size, kv_heads * head_dim, prefix + "self_attn.v_proj", precision, false, weight_fn);

    // Reshape for multi-head: [batch, seq, heads, head_dim]
    auto q_reshaped = make_multihead_reshape(q, num_heads, head_dim, prefix + "q_reshape");
    auto k_reshaped = make_multihead_reshape(k, kv_heads, head_dim, prefix + "k_reshape");
    auto v_reshaped = make_multihead_reshape(v, kv_heads, head_dim, prefix + "v_reshape");

    // Apply RoPE to Q and K (before transpose, in [batch, seq, heads, head_dim] format)
    ov::Output<ov::Node> q_for_trans = q_reshaped;
    ov::Output<ov::Node> k_for_trans = k_reshaped;
    if (position_ids.get_node() != nullptr && rope_fn) {
        q_for_trans = rope_fn(q_reshaped, position_ids, prefix + "q_rope");
        k_for_trans = rope_fn(k_reshaped, position_ids, prefix + "k_rope");
    }

    // Transpose: [batch, seq, heads, dim] -> [batch, heads, seq, dim]
    auto q_trans = make_attention_transpose(q_for_trans, prefix + "q_transpose");
    auto k_trans = make_attention_transpose(k_for_trans, prefix + "k_transpose");
    auto v_trans = make_attention_transpose(v_reshaped, prefix + "v_transpose");

    // KV cache (if enabled)
    std::vector<std::shared_ptr<ov::Node>> sinks;
    ov::Output<ov::Node> k_for_attn = k_trans;
    ov::Output<ov::Node> v_for_attn = v_trans;

    if (use_kv_cache) {
        auto layer_str = std::to_string(layer_idx);
        auto k_cache = make_kv_cache_concat(k_trans,
                                            batch_source,
                                            beam_idx,
                                            kv_heads,
                                            head_dim,
                                            "past_key_values." + layer_str + ".key" + "present." + layer_str + ".key",
                                            precision);
        auto v_cache =
            make_kv_cache_concat(v_trans,
                                 batch_source,
                                 beam_idx,
                                 kv_heads,
                                 head_dim,
                                 "past_key_values." + layer_str + ".value" + "present." + layer_str + ".value",
                                 precision);

        sinks.push_back(k_cache.assign);
        sinks.push_back(v_cache.assign);
        k_for_attn = k_cache.concatenated;
        v_for_attn = v_cache.concatenated;
    }

    // For GQA: repeat K/V heads to match Q head count
    auto k_expanded = make_repeat_kv(k_for_attn, num_heads, kv_heads, head_dim, prefix + "k_repeat");
    auto v_expanded = make_repeat_kv(v_for_attn, num_heads, kv_heads, head_dim, prefix + "v_repeat");

    // SDPA
    auto attn_output = make_sdpa(q_trans, k_expanded, v_expanded, prefix + "attn", attention_mask);

    // Transpose back and reshape
    auto attn_trans = make_attention_transpose(attn_output, prefix + "attn_out_transpose");

    auto reshape_shape = ov::opset11::Constant::create(ov::element::i64,
                                                       ov::Shape{3},
                                                       std::vector<int64_t>{0, -1, static_cast<int64_t>(hidden_size)});

    auto attn_reshaped = std::make_shared<ov::opset11::Reshape>(attn_trans, reshape_shape, true);
    attn_reshaped->set_friendly_name(prefix + "attn_reshape");

    // Output projection
    auto o_proj = make_linear(attn_reshaped->output(0),
                              hidden_size,
                              hidden_size,
                              prefix + "self_attn.o_proj",
                              precision,
                              false,
                              weight_fn);

    return {o_proj, sinks};
}

// ============================================================================
// Causal mask helper (free function)
// ============================================================================

static ov::Output<ov::Node> make_causal_mask(const ov::Output<ov::Node>& input_ids_output,
                                             const ov::Output<ov::Node>& attention_mask_output,
                                             ov::element::Type prec) {
    // --- Padding mask component: [batch, total_seq] -> [batch, 1, 1, total_seq] ---
    auto mask_float = std::make_shared<ov::opset11::Convert>(attention_mask_output, prec);
    mask_float->set_friendly_name("model.mask_convert");

    auto one_const = ov::opset11::Constant::create(prec, ov::Shape{}, {1.0f});
    auto inv_mask = std::make_shared<ov::opset11::Subtract>(one_const, mask_float);
    inv_mask->set_friendly_name("model.mask_invert");

    auto neg_inf = ov::opset11::Constant::create(prec, ov::Shape{}, {-10000.0f});
    auto padding_mask = std::make_shared<ov::opset11::Multiply>(inv_mask, neg_inf);
    padding_mask->set_friendly_name("model.padding_mask");

    auto pad_shape = ov::opset11::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 1, 1, -1});
    auto padding_4d = std::make_shared<ov::opset11::Reshape>(padding_mask, pad_shape, true);
    padding_4d->set_friendly_name("model.padding_mask_4d");

    // --- Causal mask component: [1, 1, seq_len, total_seq] ---
    auto ids_shape = std::make_shared<ov::opset11::ShapeOf>(input_ids_output, ov::element::i64);
    ids_shape->set_friendly_name("model.ids_shape");

    auto mask_shape_node = std::make_shared<ov::opset11::ShapeOf>(attention_mask_output, ov::element::i64);
    mask_shape_node->set_friendly_name("model.mask_shape");

    auto idx1 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto gather_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});

    auto seq_len_s = std::make_shared<ov::opset11::Gather>(ids_shape, idx1, gather_axis);
    seq_len_s->set_friendly_name("model.seq_len");

    auto total_seq_s = std::make_shared<ov::opset11::Gather>(mask_shape_node, idx1, gather_axis);
    total_seq_s->set_friendly_name("model.total_seq");

    auto offset = std::make_shared<ov::opset11::Subtract>(total_seq_s, seq_len_s);
    offset->set_friendly_name("model.causal_offset");

    auto range_start = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto range_step = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1});

    auto kv_range = std::make_shared<ov::op::v4::Range>(range_start, total_seq_s, range_step, ov::element::i64);
    kv_range->set_friendly_name("model.kv_range");

    auto q_range = std::make_shared<ov::op::v4::Range>(range_start, seq_len_s, range_step, ov::element::i64);
    q_range->set_friendly_name("model.q_range");

    auto q_abs = std::make_shared<ov::opset11::Add>(q_range, offset);
    q_abs->set_friendly_name("model.q_abs_positions");

    auto axis_last = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto axis_first = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {0});

    auto q_col = std::make_shared<ov::opset11::Unsqueeze>(q_abs, axis_last);
    q_col->set_friendly_name("model.q_col");

    auto kv_row = std::make_shared<ov::opset11::Unsqueeze>(kv_range, axis_first);
    kv_row->set_friendly_name("model.kv_row");

    auto causal_bool = std::make_shared<ov::op::v1::LessEqual>(kv_row, q_col);
    causal_bool->set_friendly_name("model.causal_bool");

    auto select_true = ov::opset11::Constant::create(prec, ov::Shape{}, {0.0f});
    auto select_false = ov::opset11::Constant::create(prec, ov::Shape{}, {-10000.0f});

    auto causal_float = std::make_shared<ov::op::v1::Select>(causal_bool, select_true, select_false);
    causal_float->set_friendly_name("model.causal_mask");

    auto unsqueeze_axes = ov::opset11::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});

    auto causal_4d = std::make_shared<ov::opset11::Unsqueeze>(causal_float, unsqueeze_axes);
    causal_4d->set_friendly_name("model.causal_mask_4d");

    // --- Combine: padding [batch, 1, 1, total_seq] + causal [1, 1, seq_len, total_seq] ---
    auto combined = std::make_shared<ov::opset11::Add>(padding_4d, causal_4d);
    combined->set_friendly_name("model.mask_4d");

    return combined->output(0);
}

// ============================================================================
// Simple test models (backward compatibility)
// ============================================================================

std::shared_ptr<ov::Model> ModelBuilder::get_model_with_one_op() {
    auto param = std::make_shared<ov::opset11::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
    param->set_friendly_name("input");
    auto const_value = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto add = std::make_shared<ov::opset11::Add>(param, const_value);
    add->set_friendly_name("add");
    auto result = std::make_shared<ov::opset11::Result>(add);
    result->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}

std::shared_ptr<ov::Model> ModelBuilder::get_model_without_repeated_blocks() {
    std::shared_ptr<ov::op::v0::Parameter> input =
        std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1, 1, 40});
    m_nodes.push_back(input);
    set_name(input);

    std::shared_ptr<ov::Node> res = get_block(input);

    auto result = std::make_shared<ov::op::v0::Result>(res);
    m_nodes.push_back(result);
    set_name(result);

    ov::ParameterVector params = {input};
    ov::ResultVector results = {result};

    return std::make_shared<ov::Model>(results, params);
}

std::shared_ptr<ov::Model> ModelBuilder::get_model_with_repeated_blocks(std::size_t repetitions) {
    return get_model_with_repeated_blocks_and_results(repetitions, {});
}

std::shared_ptr<ov::Model> ModelBuilder::get_model_with_repeated_blocks() {
    return get_model_with_repeated_blocks(10);
}

std::shared_ptr<ov::Model> ModelBuilder::get_model_with_repeated_blocks_and_results(
    std::size_t repetitions,
    const std::vector<std::size_t>& block_indices) {
    // Generate head
    std::shared_ptr<ov::op::v0::Parameter> input =
        std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1, 1, 40});
    m_nodes.push_back(input);
    set_name(input);

    std::vector<std::shared_ptr<ov::Node>> head(7, nullptr);
    head[0] = std::make_shared<ov::op::v1::Add>(input, input);
    head[1] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int>{2});
    head[2] = std::make_shared<ov::op::v1::Divide>(head[0], head[1], true);
    head[3] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int>{1, 1, 4, 10});
    head[4] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int>{1, 1, 40});
    head[5] = std::make_shared<ov::op::v1::Reshape>(head[2], head[3], false);
    head[6] = std::make_shared<ov::op::v1::Reshape>(head[5], head[4], false);

    for (const auto& h : head) {
        m_nodes.push_back(h);
        set_name(h);
    }

    // Generate repeated blocks
    std::shared_ptr<ov::Node> output = get_block(head[6]);
    std::vector<std::shared_ptr<ov::Node>> block_outputs;
    block_outputs.push_back(output);

    for (size_t i = 0; i < repetitions - 1; ++i) {
        output = get_block(output);
        block_outputs.push_back(output);
    }

    // Generate tail
    std::vector<std::shared_ptr<ov::Node>> tail(6, nullptr);
    tail[0] = std::make_shared<ov::op::v0::Concat>(block_outputs, -1);
    tail[1] = std::make_shared<ov::op::v0::Constant>(ov::element::i32,
                                                     ov::Shape{3},
                                                     std::vector<int>{1, 40, int(repetitions)});
    tail[2] = std::make_shared<ov::op::v1::Reshape>(tail[0], tail[1], false);
    tail[3] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1, 1, 1});
    tail[4] = std::make_shared<ov::op::v1::Multiply>(tail[2], tail[3]);
    tail[5] = std::make_shared<ov::op::v1::Add>(tail[4], tail[4]);

    for (const auto& t : tail) {
        m_nodes.push_back(t);
        set_name(t);
    }

    // Create Results
    ov::ResultVector results;

    // Add Results for specified blocks
    for (size_t idx : block_indices) {
        if (idx < block_outputs.size()) {
            auto result = std::make_shared<ov::op::v0::Result>(block_outputs[idx]);
            m_nodes.push_back(result);
            set_name(result);
            results.push_back(result);
        }
    }

    // Always add final tail Result
    auto final_result = std::make_shared<ov::op::v0::Result>(tail[5]);
    m_nodes.push_back(final_result);
    set_name(final_result);
    results.push_back(final_result);

    ov::ParameterVector params = {input};

    return std::make_shared<ov::Model>(results, params);
}

std::shared_ptr<ov::Model> ModelBuilder::get_model_with_repeated_blocks_and_parameters(
    std::size_t repetitions,
    const std::vector<std::size_t>& block_indices) {
    if (repetitions == 0) {
        repetitions = 1;
    }

    auto input = std::make_shared<ov::opset11::Parameter>(ov::element::f32, ov::Shape{1, 1, 8});
    m_nodes.push_back(input);

    ov::ParameterVector params = {input};

    std::vector<std::size_t> sorted_indices = block_indices;
    std::sort(sorted_indices.begin(), sorted_indices.end());
    sorted_indices.erase(std::unique(sorted_indices.begin(), sorted_indices.end()), sorted_indices.end());

    std::unordered_map<std::size_t, std::shared_ptr<ov::opset11::Parameter>> block_params;
    for (std::size_t idx : sorted_indices) {
        if (idx >= repetitions) {
            continue;
        }

        auto param = std::make_shared<ov::opset11::Parameter>(ov::element::f32, ov::Shape{1, 1, 8});
        m_nodes.push_back(param);
        block_params.emplace(idx, param);
        params.push_back(param);
    }

    auto scale_const = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1}, {1.f});
    auto bias_const = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1, 1, 8}, std::vector<float>(8, 0.5f));
    auto head_const = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1, 1, 8}, std::vector<float>(8, 1.f));
    m_nodes.push_back(scale_const);
    m_nodes.push_back(bias_const);
    m_nodes.push_back(head_const);

    auto head_add = std::make_shared<ov::opset11::Add>(input, head_const);
    auto head_relu = std::make_shared<ov::opset11::Relu>(head_add);
    m_nodes.push_back(head_add);
    m_nodes.push_back(head_relu);

    ov::Output<ov::Node> current = head_relu;

    for (std::size_t i = 0; i < repetitions; ++i) {
        auto it = block_params.find(i);
        ov::Output<ov::Node> rhs = (it != block_params.end()) ? it->second : current;

        auto add = std::make_shared<ov::opset11::Add>(current, rhs);
        m_nodes.push_back(add);

        auto mul = std::make_shared<ov::opset11::Multiply>(add, scale_const);
        m_nodes.push_back(mul);

        auto relu = std::make_shared<ov::opset11::Relu>(mul);
        m_nodes.push_back(relu);

        auto add_bias = std::make_shared<ov::opset11::Add>(relu, bias_const);
        m_nodes.push_back(add_bias);

        current = add_bias;
    }

    auto tail_const = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1, 1, 8}, std::vector<float>(8, 2.f));
    m_nodes.push_back(tail_const);

    auto tail_mul = std::make_shared<ov::opset11::Multiply>(current, tail_const);
    auto tail_add = std::make_shared<ov::opset11::Add>(tail_mul, tail_const);
    m_nodes.push_back(tail_mul);
    m_nodes.push_back(tail_add);

    auto result = std::make_shared<ov::opset11::Result>(tail_add);
    m_nodes.push_back(result);

    return std::make_shared<ov::Model>(ov::ResultVector{result}, params);
}

std::shared_ptr<ov::Model> ModelBuilder::get_model_with_multi_output_repeating_blocks(
    std::size_t repetitions,
    bool last_block_has_direct_result) {
    if (repetitions == 0) {
        repetitions = 1;
    }

    auto input = std::make_shared<ov::opset11::Parameter>(ov::element::f32, ov::Shape{1, 1, 8});
    m_nodes.push_back(input);
    set_name(input);

    auto add_const = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1}, {1.f});
    auto k_const = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {8});
    auto seed_indices = ov::opset11::Constant::create(ov::element::i32, ov::Shape{1, 1, 8}, {0, 1, 2, 3, 4, 5, 6, 7});
    auto tail_scale = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1}, {0.5f});
    auto tail_bias = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1}, {2.f});

    for (const auto& c : {add_const, k_const, seed_indices, tail_scale, tail_bias}) {
        m_nodes.push_back(c);
        set_name(c);
    }

    ov::Output<ov::Node> current_values = input;
    ov::Output<ov::Node> current_indices = seed_indices;

    for (std::size_t i = 0; i < repetitions; ++i) {
        auto indices_as_float = std::make_shared<ov::opset11::Convert>(current_indices, ov::element::f32);
        m_nodes.push_back(indices_as_float);
        set_name(indices_as_float);

        auto mixed = std::make_shared<ov::opset11::Add>(current_values, indices_as_float);
        m_nodes.push_back(mixed);
        set_name(mixed);

        auto shifted = std::make_shared<ov::opset11::Add>(mixed, add_const);
        m_nodes.push_back(shifted);
        set_name(shifted);

        auto topk = std::make_shared<ov::opset11::TopK>(shifted,
                                                        k_const,
                                                        -1,
                                                        ov::op::TopKMode::MAX,
                                                        ov::op::TopKSortType::SORT_VALUES,
                                                        ov::element::i32);
        m_nodes.push_back(topk);
        set_name(topk);

        current_values = topk->output(0);
        current_indices = topk->output(1);
    }

    auto tail_indices_as_float = std::make_shared<ov::opset11::Convert>(current_indices, ov::element::f32);
    m_nodes.push_back(tail_indices_as_float);
    set_name(tail_indices_as_float);

    auto tail_mixed = std::make_shared<ov::opset11::Add>(current_values, tail_indices_as_float);
    m_nodes.push_back(tail_mixed);
    set_name(tail_mixed);

    auto tail_mul = std::make_shared<ov::opset11::Multiply>(tail_mixed, tail_scale);
    m_nodes.push_back(tail_mul);
    set_name(tail_mul);

    auto tail_add = std::make_shared<ov::opset11::Add>(tail_mul, tail_bias);
    m_nodes.push_back(tail_add);
    set_name(tail_add);

    ov::ResultVector results;
    auto tail_result = std::make_shared<ov::opset11::Result>(tail_add);
    m_nodes.push_back(tail_result);
    set_name(tail_result);
    results.push_back(tail_result);

    if (last_block_has_direct_result) {
        auto direct_result = std::make_shared<ov::opset11::Result>(current_values);
        m_nodes.push_back(direct_result);
        set_name(direct_result);
        results.push_back(direct_result);
    }

    ov::ParameterVector params = {input};

    return std::make_shared<ov::Model>(results, params);
}

std::shared_ptr<ov::Node> ModelBuilder::get_block(const std::shared_ptr<ov::Node>& input) {
    std::vector<std::shared_ptr<ov::Node>> model_c(18, nullptr);
    model_c[0] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int>{0, 2, 1, 3});
    model_c[1] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int>{1});
    model_c[2] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int>{0});
    model_c[3] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int>{2});
    model_c[4] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int>{0});
    model_c[5] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int>{1, 1, 1, 1});
    model_c[6] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int>{1});
    model_c[7] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int>{0});
    model_c[8] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int>{1, 1, 1, 1});
    model_c[9] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int>{1, 1, 1, 2});
    model_c[10] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int>{1, 1, 1, 1});
    model_c[11] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int>{1, 1, 1, 2});
    model_c[12] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1, 1, 1, 1});
    model_c[13] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1, 1, 1, 1});
    model_c[14] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1, 1, 1, 1});
    model_c[15] = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{40, 40});
    model_c[16] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int>{1, 1, 4, 10});
    model_c[17] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, std::vector<int>{1, 1, 40});

    for (const auto& c : model_c) {
        m_nodes.push_back(c);
        set_name(c);
    }

    std::vector<std::shared_ptr<ov::Node>> convert(3, nullptr);
    convert[0] = std::make_shared<ov::op::v0::Convert>(model_c[15], ov::element::f16);
    convert[1] = std::make_shared<ov::op::v0::Convert>(convert[0], ov::element::i32);
    convert[2] = std::make_shared<ov::op::v0::Convert>(model_c[12], ov::element::i32);

    for (const auto& c : convert) {
        m_nodes.push_back(c);
        set_name(c);
    }

    std::vector<std::shared_ptr<ov::Node>> op(16, nullptr);
    op[0] = std::make_shared<ov::op::v0::MatMul>(input, convert[1], false, true);
    op[1] = std::make_shared<ov::op::v1::Reshape>(op[0], model_c[16], false);
    op[2] = std::make_shared<ov::op::v1::Transpose>(op[1], model_c[0]);
    op[3] = std::make_shared<ov::op::v0::ShapeOf>(op[2]);
    op[4] = std::make_shared<ov::op::v1::Gather>(op[3], model_c[1], model_c[2]);
    op[5] = std::make_shared<ov::op::v1::Divide>(op[4], model_c[3], true);
    op[6] = std::make_shared<ov::op::v0::Floor>(op[5]);
    op[7] = std::make_shared<ov::op::v3::ScatterUpdate>(model_c[5], model_c[6], op[6], model_c[7]);
    op[8] = std::make_shared<ov::op::v1::StridedSlice>(op[2],
                                                       model_c[8],
                                                       op[7],
                                                       model_c[9],
                                                       std::vector<int64_t>{1, 1, 1, 1},
                                                       std::vector<int64_t>{1, 1, 1, 1});
    op[9] = std::make_shared<ov::op::v1::StridedSlice>(op[2],
                                                       op[7],
                                                       model_c[10],
                                                       model_c[11],
                                                       std::vector<int64_t>{1, 1, 1, 1},
                                                       std::vector<int64_t>{1, 1, 1, 1});
    op[10] = std::make_shared<ov::op::v1::Multiply>(op[9], convert[2]);
    op[11] = std::make_shared<ov::op::v0::Concat>(std::vector<std::shared_ptr<ov::Node>>{op[10], op[8]}, -1);
    op[12] = std::make_shared<ov::op::v1::Multiply>(model_c[13], op[11]);
    op[13] = std::make_shared<ov::op::v1::Multiply>(model_c[14], op[2]);
    op[14] = std::make_shared<ov::op::v1::Add>(op[13], op[12]);
    op[15] = std::make_shared<ov::op::v1::Reshape>(op[14], model_c[17], false);

    for (const auto& o : op) {
        m_nodes.push_back(o);
        set_name(o);
    }

    return op[15];
}

void ModelBuilder::set_name(const std::shared_ptr<ov::Node>& node) {
    node->set_friendly_name("node_" + std::to_string(m_name_idx++));
}

// ============================================================================
// Builder Interface
// ============================================================================

std::shared_ptr<ov::op::v0::Parameter> ModelBuilder::parameter(ov::element::Type type,
                                                               const ov::PartialShape& shape,
                                                               const std::string& name) {
    auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
    param->set_friendly_name(name);
    param->output(0).set_names({name});
    m_nodes.push_back(param);
    m_parameters.push_back(param);
    return param;
}

std::shared_ptr<ov::op::v0::Result> ModelBuilder::result(const ov::Output<ov::Node>& output, const std::string& name) {
    auto res = std::make_shared<ov::op::v0::Result>(output);
    res->set_friendly_name(name);
    res->output(0).set_names({name});
    m_nodes.push_back(res);
    m_results.push_back(res);
    return res;
}

std::shared_ptr<ov::Model> ModelBuilder::build(const std::string& name) {
    return std::make_shared<ov::Model>(ov::ResultVector(m_results.begin(), m_results.end()),
                                       ov::ParameterVector(m_parameters.begin(), m_parameters.end()),
                                       name);
}

void ModelBuilder::clear() {
    m_nodes.clear();
    m_parameters.clear();
    m_results.clear();
    m_name_idx = 0;
}

// ============================================================================
// LLM Model Builder (convenience wrapper using building blocks)
// ============================================================================

std::shared_ptr<ov::Model> ModelBuilder::build_llm(const LLMConfig& config_in) {
    clear();

    // Fill default functors from scalar config values
    LLMConfig config = config_in;
    if (!config.weight)
        config.weight = FP32Weight{};
    if (!config.lm_head_weight)
        config.lm_head_weight = FP32Weight{};
    if (!config.norm)
        config.norm = LayerNorm(config.hidden_size, config.precision);
    if (config.position_ids && !config.rope)
        config.rope = HalfRotationRoPE(config.head_dim, 2048, config.precision);
    if (!config.ffn)
        config.ffn = SwiGLU(config.hidden_size, config.intermediate_size, config.precision, config.weight);

    const auto prec = config.precision;

    // ===== LLM Inputs =====
    auto attention_mask = parameter(ov::element::i64, ov::PartialShape{-1, -1}, "attention_mask");

    ov::Output<ov::Node> hidden_states;
    ov::Output<ov::Node> seq_source;  // used for causal mask seq_len and KV cache batch dim

    if (config.use_inputs_embeds) {
        auto inputs_embeds = parameter(prec,
                                       ov::PartialShape{-1, -1, static_cast<int64_t>(config.hidden_size)},
                                       "inputs_embeds");
        hidden_states = inputs_embeds->output(0);
        seq_source = inputs_embeds->output(0);
    } else {
        auto input_ids = parameter(ov::element::i64, ov::PartialShape{-1, -1}, "input_ids");
        hidden_states =
            make_embedding(input_ids->output(0), config.vocab_size, config.hidden_size, "model.embed_tokens", prec);
        seq_source = input_ids->output(0);
    }

    ov::Output<ov::Node> position_ids_output;
    if (config.position_ids) {
        position_ids_output = config.position_ids();
        // Auto-track any Parameter nodes in the position_ids subgraph
        // (may be the output itself, or upstream if the functor adds ops like Squeeze)
        std::vector<ov::Node*> stack = {position_ids_output.get_node()};
        std::set<ov::Node*> visited;
        while (!stack.empty()) {
            auto* node = stack.back();
            stack.pop_back();
            if (!visited.insert(node).second)
                continue;
            auto param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(node->shared_from_this());
            if (param) {
                m_parameters.push_back(param);
            }
            for (size_t i = 0; i < node->get_input_size(); ++i) {
                stack.push_back(node->get_input_node_ptr(i));
            }
        }
    }

    // beam_idx is required for stateful models used with LLMPipeline
    auto beam_idx = parameter(ov::element::i32, ov::PartialShape{-1}, "beam_idx");

    // ===== Attention mask =====
    auto sdpa_mask = make_causal_mask(seq_source, attention_mask->output(0), prec);

    // ===== MIDDLE: Decoder Layers =====
    ov::Output<ov::Node> current = hidden_states;
    ov::SinkVector all_sinks;

    for (size_t layer = 0; layer < config.num_layers; ++layer) {
        std::string prefix = "model.layers." + std::to_string(layer) + ".";
        SDPAttention attn{config.hidden_size,
                          config.num_heads,
                          config.get_kv_heads(),
                          config.head_dim,
                          config.precision,
                          config.weight,
                          config.rope,
                          config.use_kv_cache,
                          position_ids_output,
                          seq_source,
                          beam_idx->output(0),
                          sdpa_mask,
                          layer};
        auto layer_result = make_decoder_layer(current, config.norm, attn, config.ffn, prefix);
        current = layer_result.output;

        for (auto& sink : layer_result.sinks) {
            all_sinks.push_back(std::dynamic_pointer_cast<ov::op::Sink>(sink));
        }
    }

    // ===== TAIL: Final Norm + LM Head =====
    auto final_norm = config.norm(current, "model.norm");
    auto logits =
        make_lm_head(final_norm, config.hidden_size, config.vocab_size, "lm_head", prec, config.lm_head_weight);

    // ===== Build Model =====
    result(logits, "logits");

    auto model = std::make_shared<ov::Model>(m_results, all_sinks, m_parameters, "llm_test_model");
    return model;
}

}  // namespace npuw
}  // namespace test
}  // namespace ov
