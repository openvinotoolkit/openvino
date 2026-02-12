// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "model_builder.hpp"

#include <algorithm>
#include <set>
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
                                            const ov::Shape& shape,
                                            ov::element::Type compute_precision) const {
    // Use unique fill values per constant to prevent CSE from merging
    // different projections (e.g. Q/K/V) that happen to share dimensions.
    static size_t counter = 0;
    float fill_val = 0.01f + static_cast<float>(++counter) * 1e-7f;
    auto weight =
        ov::opset11::Constant::create(compute_precision, shape, std::vector<float>(ov::shape_size(shape), fill_val));
    weight->set_friendly_name(name);
    return weight->output(0);
}

ov::Output<ov::Node> FP16Weight::operator()(const std::string& name,
                                            const ov::Shape& shape,
                                            ov::element::Type compute_precision) const {
    static size_t counter = 0;
    float fill_val = 0.01f + static_cast<float>(++counter) * 1e-7f;
    auto weight =
        ov::opset11::Constant::create(ov::element::f16, shape, std::vector<float>(ov::shape_size(shape), fill_val));
    weight->set_friendly_name(name);

    auto convert = std::make_shared<ov::opset11::Convert>(weight, compute_precision);
    convert->set_friendly_name(name + "_convert");
    return convert->output(0);
}

ov::Output<ov::Node> CompressedWeight::operator()(const std::string& name,
                                                  const ov::Shape& shape,
                                                  ov::element::Type compute_precision) const {
    OPENVINO_ASSERT(shape.size() == 2, "CompressedWeight expects 2D shape, got ", shape.size(), "D");
    const size_t rows = shape[0];
    const size_t cols = shape[1];

    // Use unique fill values to prevent CSE from merging same-shape projections.
    // i4 range is [-8, 7], u4 is [0, 15], so clamp accordingly.
    static size_t counter = 0;
    ++counter;
    int8_t fill_val;
    if (storage_type == ov::element::i4) {
        fill_val = static_cast<int8_t>(1 + (counter % 6));  // 1..6 (within i4 range)
    } else if (storage_type == ov::element::u4) {
        fill_val = static_cast<int8_t>(1 + (counter % 14));  // 1..14 (within u4 range)
    } else {
        fill_val = static_cast<int8_t>(1 + (counter % 100));
    }
    auto weight =
        ov::opset11::Constant::create(storage_type, shape, std::vector<int8_t>(rows * cols, fill_val));
    weight->set_friendly_name(name);

    auto convert = std::make_shared<ov::opset11::Convert>(weight, ov::element::f16);
    convert->set_friendly_name(name + "_convert");

    ov::Output<ov::Node> decompressed;

    if (group_size > 0) {
        // Group quantization: reshape -> per-group scale -> reshape back
        OPENVINO_ASSERT(cols >= group_size && cols % group_size == 0,
                        "Group quantization requires cols (", cols, ") >= group_size (", group_size,
                        ") and evenly divisible");
        const size_t num_groups = cols / group_size;

        auto reshape_shape = ov::opset11::Constant::create(
            ov::element::i64,
            ov::Shape{3},
            std::vector<int64_t>{static_cast<int64_t>(rows), static_cast<int64_t>(num_groups), static_cast<int64_t>(group_size)});

        auto reshaped = std::make_shared<ov::opset11::Reshape>(convert, reshape_shape, false);
        reshaped->set_friendly_name(name + "_group_reshape");

        float scale_val = 0.01f + static_cast<float>(counter) * 1e-7f;
        auto scale = ov::opset11::Constant::create(ov::element::f16,
                                                    ov::Shape{rows, num_groups, 1},
                                                    std::vector<float>(rows * num_groups, scale_val));
        scale->set_friendly_name(name + "_scale");

        auto scaled = std::make_shared<ov::opset11::Multiply>(reshaped, scale);
        scaled->set_friendly_name(name + "_decompress");

        auto out_shape = ov::opset11::Constant::create(
            ov::element::i64,
            ov::Shape{2},
            std::vector<int64_t>{static_cast<int64_t>(rows), static_cast<int64_t>(cols)});

        auto back = std::make_shared<ov::opset11::Reshape>(scaled, out_shape, false);
        back->set_friendly_name(name + "_group_reshape_back");

        decompressed = back->output(0);
    } else {
        // Per-channel scale
        float scale_val = 0.01f + static_cast<float>(counter) * 1e-7f;
        auto scale =
            ov::opset11::Constant::create(ov::element::f16, ov::Shape{rows, 1}, std::vector<float>(rows, scale_val));
        scale->set_friendly_name(name + "_scale");

        auto scaled = std::make_shared<ov::opset11::Multiply>(convert, scale);
        scaled->set_friendly_name(name + "_decompress");

        decompressed = scaled->output(0);
    }

    if (compute_precision != ov::element::f16) {
        auto to_compute = std::make_shared<ov::opset11::Convert>(decompressed, compute_precision);
        to_compute->set_friendly_name(name + "_to_compute");
        return to_compute->output(0);
    }
    return decompressed;
}

// ============================================================================
// Norm Functor Implementations
// ============================================================================

ov::Output<ov::Node> LayerNorm::operator()(const ov::Output<ov::Node>& input, const std::string& name) const {
    static size_t counter = 0;
    float w_val = 1.0f + static_cast<float>(++counter) * 1e-7f;
    float b_val = static_cast<float>(counter) * 1e-8f;
    auto weight =
        ov::opset11::Constant::create(precision, ov::Shape{hidden_size}, std::vector<float>(hidden_size, w_val));
    weight->set_friendly_name(name + ".weight");

    auto bias = ov::opset11::Constant::create(precision, ov::Shape{hidden_size}, std::vector<float>(hidden_size, b_val));
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
    static size_t counter = 0;
    float w_val = 1.0f + static_cast<float>(++counter) * 1e-7f;
    auto weight =
        ov::opset11::Constant::create(precision, ov::Shape{hidden_size}, std::vector<float>(hidden_size, w_val));
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
// RoPE frequency builder (matches NPUW AddPositionIdsNode pattern)
// ============================================================================

/// Build frequency-based cos/sin via Sin/Cos nodes.
/// Chain: position_ids -> Unsqueeze -> Convert(f32) -> MatMul(inv_freq, .) ->
///        Transpose -> Concat(self,self) -> Sin/Cos -> Unsqueeze (broadcast over heads).
///
/// For embedding models (where position_ids comes from Range->Unsqueeze), this produces:
///   Range -> Unsqueeze -> Unsqueeze -> Convert -> MatMul -> Transpose -> Concat -> Sin/Cos
/// which matches NPUW's AddPositionIdsNode pattern exactly.
struct RoPEFrequencies {
    ov::Output<ov::Node> cos, sin;
};

static RoPEFrequencies build_rope_frequencies(size_t head_dim,
                                               ov::element::Type precision,
                                               const ov::Output<ov::Node>& position_ids,
                                               const std::string& prefix = "model.rope") {
    const size_t half_dim = head_dim / 2;

    // inv_freq = 1 / (10000 ^ (2i / head_dim)) for i = 0..half_dim-1
    // Shape: [1, half_dim, 1] for MatMul broadcasting
    std::vector<float> inv_freq_data(half_dim);
    for (size_t i = 0; i < half_dim; ++i) {
        inv_freq_data[i] = 1.0f / std::pow(10000.0f, static_cast<float>(2 * i) / static_cast<float>(head_dim));
    }
    auto inv_freq = ov::opset11::Constant::create(ov::element::f32,
                                                   ov::Shape{1, half_dim, 1},
                                                   inv_freq_data);
    inv_freq->set_friendly_name(prefix + ".inv_freq");

    // position_ids [batch, seq] or [1, seq]
    //   -> Unsqueeze(axis=1) -> [batch, 1, seq] or [1, 1, seq]
    auto unsq_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto unsqueezed = std::make_shared<ov::opset11::Unsqueeze>(position_ids, unsq_axis);
    unsqueezed->set_friendly_name(prefix + ".pos_unsqueeze");

    //   -> Convert(f32)
    auto converted = std::make_shared<ov::opset11::Convert>(unsqueezed, ov::element::f32);
    converted->set_friendly_name(prefix + ".pos_convert");

    //   -> MatMul(inv_freq [1, half_dim, 1], convert [batch, 1, seq]) = [batch, half_dim, seq]
    auto matmul = std::make_shared<ov::opset11::MatMul>(inv_freq, converted, false, false);
    matmul->set_friendly_name(prefix + ".freq_matmul");

    //   -> Transpose({0, 2, 1}) -> [batch, seq, half_dim]
    auto perm = ov::opset11::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 2, 1});
    auto transposed = std::make_shared<ov::opset11::Transpose>(matmul, perm);
    transposed->set_friendly_name(prefix + ".freq_transpose");

    //   -> Concat(self, self, axis=-1) -> [batch, seq, head_dim]
    auto concat = std::make_shared<ov::opset11::Concat>(
        ov::OutputVector{transposed->output(0), transposed->output(0)}, -1);
    concat->set_friendly_name(prefix + ".freq_concat");

    //   -> Sin / Cos
    auto sin_node = std::make_shared<ov::op::v0::Sin>(concat);
    sin_node->set_friendly_name(prefix + ".sin");
    auto cos_node = std::make_shared<ov::op::v0::Cos>(concat);
    cos_node->set_friendly_name(prefix + ".cos");

    //   -> Unsqueeze(axis=1) for broadcast over heads: [batch, 1, seq, head_dim]
    auto head_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1});

    auto cos_unsq = std::make_shared<ov::opset11::Unsqueeze>(cos_node, head_axis);
    cos_unsq->set_friendly_name(prefix + ".cos_unsqueeze");

    auto sin_unsq = std::make_shared<ov::opset11::Unsqueeze>(sin_node, head_axis);
    sin_unsq->set_friendly_name(prefix + ".sin_unsqueeze");

    // Convert to target precision if needed
    ov::Output<ov::Node> cos_out = cos_unsq->output(0);
    ov::Output<ov::Node> sin_out = sin_unsq->output(0);
    if (precision != ov::element::f32) {
        auto cos_cvt = std::make_shared<ov::opset11::Convert>(cos_unsq, precision);
        cos_cvt->set_friendly_name(prefix + ".cos_convert");
        cos_out = cos_cvt->output(0);

        auto sin_cvt = std::make_shared<ov::opset11::Convert>(sin_unsq, precision);
        sin_cvt->set_friendly_name(prefix + ".sin_convert");
        sin_out = sin_cvt->output(0);
    }

    return {cos_out, sin_out};
}

// ============================================================================
// RoPE Functor Implementations
// ============================================================================

HalfRotationRoPE::HalfRotationRoPE(size_t hd,
                                     ov::element::Type precision,
                                     const ov::Output<ov::Node>& position_ids)
    : head_dim(hd) {
    auto freq = build_rope_frequencies(hd, precision, position_ids);
    cos_freq = freq.cos;
    sin_freq = freq.sin;
}

ov::Output<ov::Node> HalfRotationRoPE::operator()(const ov::Output<ov::Node>& input,
                                                    const std::string& name) const {

    // Split into first_half and second_half along last dim
    const int64_t half = static_cast<int64_t>(head_dim / 2);

    auto zero = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto half_const = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {half});
    auto full_const = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(head_dim)});
    auto step = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto last_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {-1});

    auto first_half = std::make_shared<ov::op::v8::Slice>(input, zero, half_const, step, last_axis);
    first_half->set_friendly_name(name + "_first_half");

    auto second_half = std::make_shared<ov::op::v8::Slice>(input, half_const, full_const, step, last_axis);
    second_half->set_friendly_name(name + "_second_half");

    // rotated = Concat(-second_half, first_half)
    auto neg_one = ov::opset11::Constant::create(
        input.get_element_type(), ov::Shape{}, {-1.0f});
    auto neg_second = std::make_shared<ov::opset11::Multiply>(second_half, neg_one);
    neg_second->set_friendly_name(name + "_neg_second");

    auto rotated = std::make_shared<ov::opset11::Concat>(
        ov::OutputVector{neg_second->output(0), first_half->output(0)}, -1);
    rotated->set_friendly_name(name + "_rotated");

    // output = input * cos + rotated * sin
    auto input_cos = std::make_shared<ov::opset11::Multiply>(input, cos_freq);
    input_cos->set_friendly_name(name + "_input_cos");

    auto rotated_sin = std::make_shared<ov::opset11::Multiply>(rotated, sin_freq);
    rotated_sin->set_friendly_name(name + "_rotated_sin");

    auto output = std::make_shared<ov::opset11::Add>(input_cos, rotated_sin);
    output->set_friendly_name(name);

    return output->output(0);
}

InterleavedRoPE::InterleavedRoPE(size_t hd,
                                   ov::element::Type precision,
                                   const ov::Output<ov::Node>& position_ids)
    : head_dim(hd) {
    auto freq = build_rope_frequencies(hd, precision, position_ids);
    cos_freq = freq.cos;
    sin_freq = freq.sin;
}

ov::Output<ov::Node> InterleavedRoPE::operator()(const ov::Output<ov::Node>& input,
                                                  const std::string& name) const {

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

    auto neg_one = ov::opset11::Constant::create(input.get_element_type(), ov::Shape{}, {-1.0f});

    auto neg_x_odd = std::make_shared<ov::opset11::Multiply>(x_odd, neg_one);
    neg_x_odd->set_friendly_name(name + "_neg_x_odd");

    auto rotated_pairs = std::make_shared<ov::opset11::Concat>(ov::OutputVector{neg_x_odd, x_even}, -1);
    rotated_pairs->set_friendly_name(name + "_rotated_pairs");

    auto reshape_4d = ov::opset11::Constant::create(ov::element::i64,
                                                    ov::Shape{4},
                                                    std::vector<int64_t>{0, 0, 0, static_cast<int64_t>(head_dim)});

    auto rotated = std::make_shared<ov::opset11::Reshape>(rotated_pairs, reshape_4d, true);
    rotated->set_friendly_name(name + "_rotated");

    auto input_cos = std::make_shared<ov::opset11::Multiply>(input, cos_freq);
    input_cos->set_friendly_name(name + "_input_cos");

    auto rotated_sin = std::make_shared<ov::opset11::Multiply>(rotated, sin_freq);
    rotated_sin->set_friendly_name(name + "_rotated_sin");

    auto output = std::make_shared<ov::opset11::Add>(input_cos, rotated_sin);
    output->set_friendly_name(name);

    return output->output(0);
}

// ============================================================================
// Position IDs helpers
// ============================================================================

ov::Output<ov::Node> make_position_ids_2d() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
    param->set_friendly_name("position_ids");
    param->output(0).set_names({"position_ids"});
    return param->output(0);
}

ov::Output<ov::Node> make_position_ids_3d() {
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
    auto weight_output = weight_fn(name + ".weight", ov::Shape{out_features, in_features}, precision);

    auto matmul = std::make_shared<ov::opset11::MatMul>(input, weight_output, false, true);
    matmul->set_friendly_name(name);

    if (add_bias) {
        static size_t bias_counter = 0;
        float bias_val = static_cast<float>(++bias_counter) * 1e-8f;
        auto bias =
            ov::opset11::Constant::create(precision, ov::Shape{out_features}, std::vector<float>(out_features, bias_val));
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
                                    const std::string& name,
                                    const ov::Output<ov::Node>& shared_broadcast_shape) {
    const size_t actual_kv_heads = (num_kv_heads == 0) ? num_heads : num_kv_heads;
    const size_t n_rep = num_heads / actual_kv_heads;

    // Skip expansion only when no shared shape and MHA (n_rep=1)
    if (!shared_broadcast_shape.get_node() && n_rep == 1) {
        return kv;
    }

    OPENVINO_ASSERT(num_heads % actual_kv_heads == 0,
                    "num_heads (", num_heads, ") must be divisible by num_kv_heads (", actual_kv_heads, ")");

    // Use shared shape if provided, otherwise compute per-layer
    ov::Output<ov::Node> broadcast_shape_output;
    if (shared_broadcast_shape.get_node()) {
        broadcast_shape_output = shared_broadcast_shape;
    } else {
        auto shape_of_kv = std::make_shared<ov::opset11::ShapeOf>(kv, ov::element::i64);
        auto gather_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});
        auto idx_01 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
        auto batch_kv_heads = std::make_shared<ov::opset11::Gather>(shape_of_kv, idx_01, gather_axis);
        auto n_rep_const =
            ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(n_rep)});
        auto idx_23 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{2}, {2, 3});
        auto seq_head_dim = std::make_shared<ov::opset11::Gather>(shape_of_kv, idx_23, gather_axis);
        auto broadcast_shape =
            std::make_shared<ov::opset11::Concat>(ov::OutputVector{batch_kv_heads, n_rep_const, seq_head_dim}, 0);
        broadcast_shape->set_friendly_name(name + "_broadcast_shape");
        broadcast_shape_output = broadcast_shape->output(0);
    }

    auto unsqueeze_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {2});

    auto unsqueezed = std::make_shared<ov::opset11::Unsqueeze>(kv, unsqueeze_axis);
    unsqueezed->set_friendly_name(name + "_unsqueeze");

    auto broadcasted = std::make_shared<ov::op::v3::Broadcast>(unsqueezed,
                                                                broadcast_shape_output,
                                                                ov::op::BroadcastType::BIDIRECTIONAL);
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

KVCacheReadState make_kv_cache_read(const ov::Output<ov::Node>& batch_source,
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

    return {variable, beam_gather->output(0)};
}

KVCacheResult make_kv_cache_concat(const ov::Output<ov::Node>& current_kv,
                                   const ov::Output<ov::Node>& batch_source,
                                   const ov::Output<ov::Node>& beam_idx,
                                   size_t num_heads,
                                   size_t head_dim,
                                   const std::string& name,
                                   ov::element::Type precision) {
    auto read_state = make_kv_cache_read(batch_source, beam_idx, num_heads, head_dim, name, precision);

    auto concat = std::make_shared<ov::opset11::Concat>(
        ov::OutputVector{read_state.beam_gather, current_kv}, 2);
    concat->set_friendly_name(name + "_concat");

    auto assign = std::make_shared<ov::op::v6::Assign>(concat, read_state.variable);
    assign->set_friendly_name(name + "_assign");

    return {concat->output(0), read_state.beam_gather, assign};
}

ov::Output<ov::Node> make_sdpa(const ov::Output<ov::Node>& q,
                               const ov::Output<ov::Node>& k,
                               const ov::Output<ov::Node>& v,
                               const std::string& name,
                               const ov::Output<ov::Node>& attention_mask,
                               size_t head_dim_for_scale) {
    std::shared_ptr<ov::op::v13::ScaledDotProductAttention> sdpa;
    if (head_dim_for_scale > 0 && attention_mask.get_node()) {
        // 5-input SDPA: Q, K, V, mask, scale (required for embedding model pattern matching)
        auto scale_val = 1.0f / std::sqrt(static_cast<float>(head_dim_for_scale));
        auto scale = ov::opset11::Constant::create(ov::element::f32, ov::Shape{}, {scale_val});
        scale->set_friendly_name(name + ".scale");
        sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, attention_mask, scale, false);
    } else if (attention_mask.get_node()) {
        sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, attention_mask, false);
    } else {
        sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, false);
    }
    sdpa->set_friendly_name(name + ".sdpa");

    return sdpa->output(0);
}

ov::Output<ov::Node> make_attention_output(const ov::Output<ov::Node>& sdpa_output,
                                           size_t hidden_size,
                                           const std::string& name,
                                           ov::element::Type precision,
                                           bool add_bias,
                                           const WeightFn& weight_fn) {
    auto attn_trans = make_attention_transpose(sdpa_output, name + "_transpose");

    auto reshape_shape = ov::opset11::Constant::create(
        ov::element::i64, ov::Shape{3},
        std::vector<int64_t>{0, -1, static_cast<int64_t>(hidden_size)});
    auto attn_reshaped = std::make_shared<ov::opset11::Reshape>(attn_trans, reshape_shape, true);
    attn_reshaped->set_friendly_name(name + "_reshape");

    return make_linear(attn_reshaped->output(0), hidden_size, hidden_size, name, precision, add_bias, weight_fn);
}

ov::Output<ov::Node> make_embedding(const ov::Output<ov::Node>& input_ids,
                                    size_t vocab_size,
                                    size_t hidden_size,
                                    const std::string& name,
                                    ov::element::Type precision) {
    static size_t counter = 0;
    float fill_val = 0.01f + static_cast<float>(++counter) * 1e-7f;
    auto weight = ov::opset11::Constant::create(precision,
                                                ov::Shape{vocab_size, hidden_size},
                                                std::vector<float>(vocab_size * hidden_size, fill_val));
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

ov::Output<ov::Node> make_conv1d(const ov::Output<ov::Node>& input,
                                 size_t in_channels,
                                 size_t out_channels,
                                 size_t kernel_size,
                                 size_t stride,
                                 size_t padding,
                                 const std::string& name,
                                 ov::element::Type precision) {
    static size_t counter = 0;
    float w_val = 0.01f + static_cast<float>(++counter) * 1e-7f;
    float b_val = static_cast<float>(counter) * 1e-8f;
    auto weight = ov::opset11::Constant::create(
        precision,
        ov::Shape{out_channels, in_channels, kernel_size},
        std::vector<float>(out_channels * in_channels * kernel_size, w_val));
    weight->set_friendly_name(name + ".weight");

    auto conv = std::make_shared<ov::op::v1::Convolution>(
        input,
        weight,
        ov::Strides{stride},
        ov::CoordinateDiff{static_cast<std::ptrdiff_t>(padding)},
        ov::CoordinateDiff{static_cast<std::ptrdiff_t>(padding)},
        ov::Strides{1});
    conv->set_friendly_name(name);

    auto bias = ov::opset11::Constant::create(
        precision,
        ov::Shape{1, out_channels, 1},
        std::vector<float>(out_channels, b_val));
    bias->set_friendly_name(name + ".bias");

    auto add = std::make_shared<ov::opset11::Add>(conv, bias);
    add->set_friendly_name(name + "_bias_add");

    return add->output(0);
}

KVCacheResult make_encoder_kv_cache(const ov::Output<ov::Node>& encoder_kv,
                                    size_t num_heads,
                                    size_t head_dim,
                                    const std::string& name,
                                    ov::element::Type precision) {
    auto var_shape = ov::PartialShape{-1, static_cast<int64_t>(num_heads), -1, static_cast<int64_t>(head_dim)};
    auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{var_shape, precision, name});

    // ReadValue initialized from projected encoder K/V
    // Output goes directly to SDPA and Assign (NO Gather/beam reorder —
    // encoder KV is identical across beams, matching the real whisper model pattern).
    auto read_value = std::make_shared<ov::op::v6::ReadValue>(encoder_kv, variable);
    read_value->set_friendly_name(name + "_read");

    // Assign directly from ReadValue (no concat, no beam reorder)
    auto assign = std::make_shared<ov::op::v6::Assign>(read_value, variable);
    assign->set_friendly_name(name + "_assign");

    return {read_value->output(0), {}, assign};
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
    auto up = make_linear(input, hidden_size, intermediate_size, name + ".up_proj", precision, use_bias, weight_fn);

    auto gelu = std::make_shared<ov::opset11::Gelu>(up);
    gelu->set_friendly_name(name + "_gelu");

    auto down = make_linear(gelu, intermediate_size, hidden_size, name + ".down_proj", precision, use_bias, weight_fn);

    return down;
}

// ============================================================================
// KV cache variable ID helper
// ============================================================================

/// Produces concatenated variable_id for KV cache state variables.
/// The missing separator (e.g. "keypresent") is intentional — matches
/// OV's StatefulToStateless pass which concatenates input+output names.
/// @param layer   Layer index as string (e.g. "0", "1")
/// @param infix   "." for LLM, ".decoder." or ".encoder." for whisper
/// @param kv_type "key" or "value"
static std::string make_kv_var_id(const std::string& layer,
                                  const std::string& infix,
                                  const std::string& kv_type) {
    return "past_key_values." + layer + infix + kv_type
         + "present." + layer + infix + kv_type;
}

// ============================================================================
// Attention Functor Implementation
// ============================================================================

LayerResult Attention::operator()(const ov::Output<ov::Node>& input, const std::string& prefix) const {
    // K/V source: self-attention uses input, cross-attention uses kv_source
    auto kv_input = kv_source.get_node() ? kv_source : input;

    // Derive attention-type prefix from k_proj_name for unique intermediate node names.
    // e.g. "self_attn.k_proj" -> "self_attn.", "encoder_attn.k_proj" -> "encoder_attn."
    auto attn_prefix = k_proj_name.substr(0, k_proj_name.rfind('.') + 1);

    // Q, K, V projections
    auto q = make_linear(input, hidden_size, num_heads * head_dim,
                         prefix + q_proj_name, precision, add_bias, weight_fn);
    auto k = make_linear(kv_input, hidden_size, num_kv_heads * head_dim,
                         prefix + k_proj_name, precision, add_bias, weight_fn);
    auto v = make_linear(kv_input, hidden_size, num_kv_heads * head_dim,
                         prefix + v_proj_name, precision, add_bias, weight_fn);

    // Reshape for multi-head: [batch, seq, heads, head_dim]
    auto q_reshaped = make_multihead_reshape(q, num_heads, head_dim, prefix + attn_prefix + "q_reshape");
    auto k_reshaped = make_multihead_reshape(k, num_kv_heads, head_dim, prefix + attn_prefix + "k_reshape");
    auto v_reshaped = make_multihead_reshape(v, num_kv_heads, head_dim, prefix + attn_prefix + "v_reshape");

    // Optional QK-norm: applied to Q and K after reshape, before RoPE
    ov::Output<ov::Node> q_normed = q_reshaped;
    ov::Output<ov::Node> k_normed = k_reshaped;
    if (qk_norm) {
        q_normed = qk_norm(q_reshaped, prefix + attn_prefix + "q_norm");
        k_normed = qk_norm(k_reshaped, prefix + attn_prefix + "k_norm");
    }

    // Transpose first: [batch, seq, heads, dim] -> [batch, heads, seq, dim]
    auto q_trans = make_attention_transpose(q_normed, prefix + attn_prefix + "q_transpose");
    auto k_trans = make_attention_transpose(k_normed, prefix + attn_prefix + "k_transpose");
    auto v_trans = make_attention_transpose(v_reshaped, prefix + attn_prefix + "v_transpose");

    // Apply RoPE to Q and K (after transpose, on [batch, heads, seq, dim])
    ov::Output<ov::Node> q_roped = q_trans;
    ov::Output<ov::Node> k_roped = k_trans;
    if (rope_fn) {
        q_roped = rope_fn(q_trans, prefix + "q_rope");
        k_roped = rope_fn(k_trans, prefix + "k_rope");
    }

    // KV cache
    std::vector<std::shared_ptr<ov::Node>> sinks;
    ov::Output<ov::Node> k_for_attn = k_roped;
    ov::Output<ov::Node> v_for_attn = v_trans;

    if (cache_mode == CacheMode::ConcatBeam) {
        auto layer_str = std::to_string(layer_idx);
        auto k_var_id = make_kv_var_id(layer_str, cache_infix, "key");

        KVCacheResult k_cache;
        if (prebuilt_k_variable) {
            // Reuse pre-built Variable/ReadValue/beam_gather (layer 0 key cache)
            auto k_concat = std::make_shared<ov::opset11::Concat>(
                ov::OutputVector{prebuilt_k_beam_gather, k_roped}, 2);
            k_concat->set_friendly_name(k_var_id + "_concat");
            auto k_assign = std::make_shared<ov::op::v6::Assign>(k_concat, prebuilt_k_variable);
            k_assign->set_friendly_name(k_var_id + "_assign");
            k_cache = {k_concat->output(0), prebuilt_k_beam_gather, k_assign};
        } else {
            k_cache = make_kv_cache_concat(
                k_roped, batch_source, beam_idx, num_kv_heads, head_dim, k_var_id, precision);
        }

        auto v_cache = make_kv_cache_concat(
            v_trans, batch_source, beam_idx, num_kv_heads, head_dim,
            make_kv_var_id(layer_str, cache_infix, "value"), precision);

        sinks.push_back(k_cache.assign);
        sinks.push_back(v_cache.assign);
        k_for_attn = k_cache.concatenated;
        v_for_attn = v_cache.concatenated;
    } else if (cache_mode == CacheMode::StoreOnly) {
        auto layer_str = std::to_string(layer_idx);
        auto k_cache = make_encoder_kv_cache(
            k_roped, num_kv_heads, head_dim,
            make_kv_var_id(layer_str, cache_infix, "key"), precision);
        auto v_cache = make_encoder_kv_cache(
            v_trans, num_kv_heads, head_dim,
            make_kv_var_id(layer_str, cache_infix, "value"), precision);

        sinks = {k_cache.assign, v_cache.assign};
        k_for_attn = k_cache.concatenated;
        v_for_attn = v_cache.concatenated;
    }

    // For GQA: repeat K/V heads to match Q head count
    auto k_expanded = make_repeat_kv(k_for_attn, num_heads, num_kv_heads, head_dim, prefix + "k_repeat",
                                     shared_broadcast_shape);
    auto v_expanded = make_repeat_kv(v_for_attn, num_heads, num_kv_heads, head_dim, prefix + "v_repeat",
                                     shared_broadcast_shape);

    // SDPA
    // Pass head_dim for scale when in embedding mode (shared_broadcast_shape set)
    // to create 5-input SDPA matching ReConstructEmbeddingModel pattern
    size_t sdpa_scale_dim = shared_broadcast_shape.get_node() ? head_dim : 0;
    auto attn_output = make_sdpa(q_roped, k_expanded, v_expanded, prefix + "self_attn.attn", sdpa_mask, sdpa_scale_dim);

    auto o_proj = make_attention_output(attn_output, hidden_size,
                                        prefix + o_proj_name, precision, add_bias, weight_fn);

    return {o_proj, sinks};
}

// ============================================================================
// Causal mask helper (free function)
// ============================================================================

/// Padding-only mask: [batch, seq] -> [batch, 1, 1, seq] float (0.0=attend, -10000.0=pad)
static ov::Output<ov::Node> make_padding_mask(const ov::Output<ov::Node>& attention_mask_output,
                                              ov::element::Type prec) {
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

    return padding_4d->output(0);
}

static ov::Output<ov::Node> make_causal_mask(const ov::Output<ov::Node>& input_ids_output,
                                             const ov::Output<ov::Node>& attention_mask_output,
                                             ov::element::Type prec) {
    // --- Padding mask component: [batch, 1, 1, total_seq] ---
    auto padding_4d = make_padding_mask(attention_mask_output, prec);

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
    clear();
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
    clear();
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
    clear();
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
    clear();
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
    m_parameters.push_back(param);
    return param;
}

std::shared_ptr<ov::op::v0::Result> ModelBuilder::result(const ov::Output<ov::Node>& output, const std::string& name) {
    auto res = std::make_shared<ov::op::v0::Result>(output);
    res->set_friendly_name(name);
    res->output(0).set_names({name});
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

    // ===== Position IDs =====
    ov::Output<ov::Node> position_ids_output;
    if (config.internal_position_ids) {
        // Generate position_ids internally from input shape: arange(0, seq_len) -> [1, seq]
        auto shape = std::make_shared<ov::opset11::ShapeOf>(seq_source, ov::element::i64);
        auto idx1 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1});
        auto axis0 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});
        auto seq_len = std::make_shared<ov::opset11::Gather>(shape, idx1, axis0);
        auto start = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});
        auto step = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1});
        auto range = std::make_shared<ov::op::v4::Range>(start, seq_len, step, ov::element::i64);
        range->set_friendly_name("model.internal_position_ids_range");
        auto unsqueeze_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto unsqueezed = std::make_shared<ov::opset11::Unsqueeze>(range, unsqueeze_axis);
        unsqueezed->set_friendly_name("model.internal_position_ids");
        position_ids_output = unsqueezed->output(0);
        // No disconnected position_ids Parameter needed: the frequency-based RoPE chain
        // (Range -> Unsqueeze -> Unsqueeze -> Convert -> MatMul -> ... -> Sin/Cos) matches
        // NPUW's AddPositionIdsNode pattern, which creates a connected position_ids Parameter.
    } else if (config.position_ids.get_node()) {
        // User provided custom position_ids (3D, Range subgraph, etc.)
        position_ids_output = config.position_ids;
        // Auto-track any Parameter nodes in the position_ids subgraph
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
    } else if (!config.rope) {
        // No explicit position_ids and no explicit rope: auto-create 2D Parameter
        // (default LLM behavior — standard position_ids input)
        position_ids_output = make_position_ids_2d();
        // Track the auto-created Parameter
        auto param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(
            position_ids_output.get_node_shared_ptr());
        if (param) {
            m_parameters.push_back(param);
        }
    }
    // else: config.rope is set but no position_ids — RoPE was pre-built with position_ids baked in

    // Build default RoPE if position_ids are available but rope is not yet set
    if (position_ids_output.get_node() && !config.rope) {
        config.rope = HalfRotationRoPE(config.head_dim, config.precision, position_ids_output);
    }

    // beam_idx is required for stateful models used with LLMPipeline
    ov::Output<ov::Node> beam_idx_output;
    if (config.use_kv_cache) {
        auto beam_idx = parameter(ov::element::i32, ov::PartialShape{-1}, "beam_idx");
        beam_idx_output = beam_idx->output(0);
    }

    // ===== Attention mask =====
    auto sdpa_mask = make_causal_mask(seq_source, attention_mask->output(0), prec);

    // ===== Shared GQA broadcast shape for embedding models =====
    // ReConstructEmbeddingModel requires all SDPA nodes share the same Broadcast shape Concat.
    // Build it from attention_mask shape (available to all layers) with 5 individual inputs
    // so update_kv_concat_shape can replace input(3) = seq with mask-based seq_len.
    ov::Output<ov::Node> shared_broadcast;
    if (!config.use_kv_cache && !config.use_lm_head) {
        const size_t kv_heads = config.get_kv_heads();
        const size_t n_rep = config.num_heads / kv_heads;
        auto shape_of_mask = std::make_shared<ov::opset11::ShapeOf>(attention_mask->output(0), ov::element::i64);
        auto axis0 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});
        auto idx0 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto batch_dim = std::make_shared<ov::opset11::Gather>(shape_of_mask, idx0, axis0);
        auto idx1 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        auto seq_dim = std::make_shared<ov::opset11::Gather>(shape_of_mask, idx1, axis0);
        auto kv_heads_const = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1},
                                                             {static_cast<int64_t>(kv_heads)});
        auto n_rep_const = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1},
                                                          {static_cast<int64_t>(n_rep)});
        auto head_dim_const = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1},
                                                             {static_cast<int64_t>(config.head_dim)});
        auto shared_concat = std::make_shared<ov::opset11::Concat>(
            ov::OutputVector{batch_dim, kv_heads_const, n_rep_const, seq_dim, head_dim_const}, 0);
        shared_concat->set_friendly_name("model.shared_gqa_broadcast_shape");
        shared_broadcast = shared_concat->output(0);
    }

    // ===== MIDDLE: Decoder Layers =====
    ov::Output<ov::Node> current = hidden_states;
    ov::SinkVector all_sinks;

    for (size_t layer = 0; layer < config.num_layers; ++layer) {
        std::string prefix = "model.layers." + std::to_string(layer) + ".";
        Attention attn{config.hidden_size, config.num_heads, config.get_kv_heads(),
                       config.head_dim, config.precision, config.weight};
        attn.qk_norm = config.qk_norm;
        attn.rope_fn = config.rope;
        attn.cache_mode = config.use_kv_cache ? Attention::CacheMode::ConcatBeam : Attention::CacheMode::None;
        attn.batch_source = seq_source;
        attn.beam_idx = beam_idx_output;
        attn.layer_idx = layer;
        attn.sdpa_mask = sdpa_mask;
        attn.shared_broadcast_shape = shared_broadcast;
        auto layer_result = make_decoder_layer(current, config.norm, attn, config.ffn, prefix);
        current = layer_result.output;

        for (auto& sink : layer_result.sinks) {
            all_sinks.push_back(std::dynamic_pointer_cast<ov::op::Sink>(sink));
        }
    }

    // ===== TAIL: Final Norm + optional LM Head =====
    auto final_norm = config.norm(current, "model.norm");

    if (config.use_lm_head) {
        auto logits =
            make_lm_head(final_norm, config.hidden_size, config.vocab_size, "lm_head", prec, config.lm_head_weight);
        result(logits, "logits");
    } else {
        result(final_norm, "last_hidden_state");
    }

    // ===== Build Model =====
    auto model = std::make_shared<ov::Model>(m_results, all_sinks, m_parameters, "llm_test_model");
    return model;
}

// ============================================================================
// Whisper Encoder Builder
// ============================================================================

std::shared_ptr<ov::Model> ModelBuilder::build_whisper_encoder(const WhisperConfig& config) {
    clear();

    const auto prec = config.precision;
    const auto d = config.d_model;
    const auto heads = config.encoder_attention_heads;
    const auto hd = config.head_dim();
    WeightFn wf = config.weight ? config.weight : WeightFn{FP32Weight{}};

    // Input: [batch, num_mel_bins, 3000] — always f32 (audio features from feature extractor)
    auto input_features = parameter(ov::element::f32,
        ov::PartialShape{-1, static_cast<int64_t>(config.num_mel_bins), 3000},
        "input_features");

    // Convert to internal precision if needed (convolution requires matching types)
    ov::Output<ov::Node> encoder_input = input_features->output(0);
    if (prec != ov::element::f32) {
        auto cvt = std::make_shared<ov::op::v0::Convert>(encoder_input, prec);
        cvt->set_friendly_name("model.encoder.input_convert");
        encoder_input = cvt->output(0);
    }

    // Conv1: (num_mel_bins -> d_model, kernel=3, stride=1, padding=1)
    auto conv1 = make_conv1d(encoder_input, config.num_mel_bins, d, 3, 1, 1,
                             "model.encoder.conv1", prec);
    auto gelu1 = std::make_shared<ov::opset11::Gelu>(conv1);
    gelu1->set_friendly_name("model.encoder.conv1_gelu");

    // Conv2: (d_model -> d_model, kernel=3, stride=2, padding=1)
    auto conv2 = make_conv1d(gelu1->output(0), d, d, 3, 2, 1, "model.encoder.conv2", prec);
    auto gelu2 = std::make_shared<ov::opset11::Gelu>(conv2);
    gelu2->set_friendly_name("model.encoder.conv2_gelu");

    // Transpose: [1, d_model, max_source_positions] -> [1, max_source_positions, d_model]
    auto transpose_order = ov::opset11::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
    auto transposed = std::make_shared<ov::opset11::Transpose>(gelu2, transpose_order);
    transposed->set_friendly_name("model.encoder.transpose");

    // Positional embedding: learned constant [1, max_source_positions, d_model]
    auto pos_embed = ov::opset11::Constant::create(prec,
        ov::Shape{1, config.max_source_positions, d},
        std::vector<float>(config.max_source_positions * d, 0.02f));
    pos_embed->set_friendly_name("model.encoder.embed_positions.weight");

    auto embedded = std::make_shared<ov::opset11::Add>(transposed, pos_embed);
    embedded->set_friendly_name("model.encoder.pos_embed_add");

    // Encoder layers: same 2-sublayer structure as LLM (norm -> self_attn -> residual -> norm -> FFN -> residual)
    LayerNorm norm(d, prec);
    GELUFn ffn(d, config.encoder_ffn_dim, prec, wf, true);  // bias=true

    ov::Output<ov::Node> current = embedded->output(0);

    for (size_t layer = 0; layer < config.encoder_layers; ++layer) {
        std::string prefix = "model.encoder.layers." + std::to_string(layer) + ".";

        Attention self_attn{d, heads, heads, hd, prec, wf};
        self_attn.add_bias = true;
        self_attn.layer_idx = layer;
        self_attn.o_proj_name = "self_attn.out_proj";

        auto layer_result = make_decoder_layer(current, norm, self_attn, ffn, prefix);
        current = layer_result.output;
    }

    // Final LayerNorm
    auto final_norm = norm(current, "model.encoder.layer_norm");

    // Result — always f32 output (WhisperPipeline reads encoder output as f32)
    ov::Output<ov::Node> encoder_output = final_norm;
    if (prec != ov::element::f32) {
        auto cvt = std::make_shared<ov::op::v0::Convert>(final_norm, ov::element::f32);
        cvt->set_friendly_name("model.encoder.output_convert");
        encoder_output = cvt->output(0);
    }
    result(encoder_output, "last_hidden_state");

    return std::make_shared<ov::Model>(
        ov::ResultVector(m_results.begin(), m_results.end()),
        ov::ParameterVector(m_parameters.begin(), m_parameters.end()),
        "whisper_encoder");
}

// ============================================================================
// Whisper Decoder Builder
// ============================================================================

std::shared_ptr<ov::Model> ModelBuilder::build_whisper_decoder(const WhisperConfig& config) {
    clear();

    const auto prec = config.precision;
    const auto d = config.d_model;
    const auto heads = config.decoder_attention_heads;
    const auto hd = config.head_dim();
    WeightFn wf = config.weight ? config.weight : WeightFn{FP32Weight{}};

    // Inputs — encoder_hidden_states is always f32 (matches encoder output)
    auto input_ids = parameter(ov::element::i64, ov::PartialShape{-1, -1}, "input_ids");
    auto encoder_hidden_states = parameter(ov::element::f32,
        ov::PartialShape{-1, -1, static_cast<int64_t>(d)},
        "encoder_hidden_states");
    auto beam_idx = parameter(ov::element::i32, ov::PartialShape{-1}, "beam_idx");

    // Convert encoder_hidden_states to internal precision if needed
    ov::Output<ov::Node> enc_hs = encoder_hidden_states->output(0);
    if (prec != ov::element::f32) {
        auto cvt = std::make_shared<ov::op::v0::Convert>(enc_hs, prec);
        cvt->set_friendly_name("model.decoder.enc_hs_convert");
        enc_hs = cvt->output(0);
    }

    // Token embedding
    auto token_embed = make_embedding(input_ids->output(0), config.vocab_size, d, "model.decoder.embed_tokens", prec);

    // seq_len = ShapeOf(input_ids)[1]
    auto ids_shape = std::make_shared<ov::opset11::ShapeOf>(input_ids, ov::element::i64);
    ids_shape->set_friendly_name("model.decoder.ids_shape");
    auto seq_len = std::make_shared<ov::opset11::Gather>(
        ids_shape,
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1}),
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    seq_len->set_friendly_name("model.decoder.seq_len");

    // ================================================================
    // Pre-build layer 0's key KV cache READ state to derive kv_seq_len.
    // The CachePositionInput and AttentionMaskInputPast_2 NPUW matchers
    // need kv_seq_len from ShapeOf(beam_gather)[2] to construct patterns
    // that match the real whisper model's graph structure.
    // ================================================================
    auto layer0_k_read = make_kv_cache_read(
        input_ids->output(0), beam_idx->output(0), heads, hd,
        make_kv_var_id("0", ".decoder.", "key"), prec);

    // kv_seq_len = ShapeOf(beam_gather)[2]
    // This Gather is the ROOT of the CachePositionInput pattern.
    auto kv_shape = std::make_shared<ov::opset11::ShapeOf>(layer0_k_read.beam_gather, ov::element::i64);
    kv_shape->set_friendly_name("model.decoder.kv_shape");
    auto kv_seq_len = std::make_shared<ov::opset11::Gather>(
        kv_shape,
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {2}),
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    kv_seq_len->set_friendly_name("model.decoder.kv_seq_len");

    // ================================================================
    // CachePositionInput pattern:
    // Gather(kv_seq_len) → Add(seq_len) → Range(Gather, Add, 1) → Unsqueeze → Tile
    // NPUW replaces the Range with cache_position Parameter in kvcache model.
    // ================================================================
    auto total_seq_len = std::make_shared<ov::opset11::Add>(kv_seq_len, seq_len);
    total_seq_len->set_friendly_name("model.decoder.total_seq_len");

    auto cache_positions = std::make_shared<ov::op::v4::Range>(
        kv_seq_len->output(0),
        total_seq_len->output(0),
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1})->output(0),
        ov::element::i64);
    cache_positions->set_friendly_name("model.decoder.cache_positions");

    auto cache_pos_unsq = std::make_shared<ov::opset11::Unsqueeze>(
        cache_positions,
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    cache_pos_unsq->set_friendly_name("model.decoder.cache_pos_unsq");

    // Tile repeats: [batch_dim, 1]
    auto batch_dim_for_tile = std::make_shared<ov::opset11::Gather>(
        ids_shape,
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {0}),
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    batch_dim_for_tile->set_friendly_name("model.decoder.batch_for_tile");

    auto tile_repeats = std::make_shared<ov::opset11::Concat>(
        ov::OutputVector{
            batch_dim_for_tile->output(0),
            ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1})->output(0)},
        0);
    tile_repeats->set_friendly_name("model.decoder.tile_repeats");

    auto position_ids = std::make_shared<ov::op::v0::Tile>(cache_pos_unsq, tile_repeats);
    position_ids->set_friendly_name("model.decoder.position_ids");

    // ================================================================
    // Positional embedding: Gather(pos_embed_table, position_ids, 0)
    // Uses Tile output (position_ids) so that after CachePositionInput
    // matching, the lookup correctly uses the cache_position parameter.
    // ================================================================
    static size_t pos_counter = 0;
    float pos_fill = 0.03f + static_cast<float>(++pos_counter) * 1e-7f;
    auto pos_embed_table = ov::opset11::Constant::create(prec,
        ov::Shape{config.max_target_positions, d},
        std::vector<float>(config.max_target_positions * d, pos_fill));
    pos_embed_table->set_friendly_name("model.decoder.embed_positions.weight");

    // Convert i64 → i32 for Gather index (matches real whisper model)
    auto pos_ids_i32 = std::make_shared<ov::op::v0::Convert>(position_ids, ov::element::i32);
    pos_ids_i32->set_friendly_name("model.decoder.pos_ids_convert");
    auto pos_embed = std::make_shared<ov::opset11::Gather>(
        pos_embed_table, pos_ids_i32,
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0}), 0);
    pos_embed->set_friendly_name("model.decoder.pos_embed_gather");

    auto hidden_states = std::make_shared<ov::opset11::Add>(token_embed, pos_embed);
    hidden_states->set_friendly_name("model.decoder.embed_add");

    // ================================================================
    // Causal mask: AttentionMaskInputPast_2 pattern + Slice for prefill
    //
    // Pattern: Range(0, total_seq, 1) → Unsqueeze × 3 → LessEqual(kv_idx, q_idx)
    // Then: Broadcast → Select(0.0, -inf) → Slice(axis=3) → SDPA input[3]
    //
    // AttentionMaskInputPast_2 matches on the 3-unsqueeze chain and replaces
    // LessEqual with an attention_mask Parameter.
    // AttentionMaskInput (prefill) matches on Slice → SDPA input[3].
    // ================================================================
    ov::Output<ov::Node> shared_mask;
    {
        // kv_idx: Range(0, total_seq, 1) → Unsqueeze(0) → Unsqueeze(1) → Unsqueeze(2)
        // gives shape [1, 1, 1, total_seq]
        auto mask_range = std::make_shared<ov::op::v4::Range>(
            ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0})->output(0),
            total_seq_len->output(0),
            ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1})->output(0),
            ov::element::i64);
        mask_range->set_friendly_name("model.decoder.mask_range");

        auto kv_unsq1 = std::make_shared<ov::opset11::Unsqueeze>(
            mask_range,
            ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0}));
        kv_unsq1->set_friendly_name("model.decoder.kv_unsq1");
        auto kv_unsq2 = std::make_shared<ov::opset11::Unsqueeze>(
            kv_unsq1,
            ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1}));
        kv_unsq2->set_friendly_name("model.decoder.kv_unsq2");
        auto kv_unsq3 = std::make_shared<ov::opset11::Unsqueeze>(
            kv_unsq2,
            ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {2}));
        kv_unsq3->set_friendly_name("model.decoder.kv_unsq3");
        // kv_unsq3 shape: [1, 1, 1, total_seq]

        // q_idx: cache_pos_unsq [1, seq] → Unsqueeze(1) → [1, 1, seq]
        //        → Unsqueeze(3) → [1, 1, seq, 1]
        auto q_unsq1 = std::make_shared<ov::opset11::Unsqueeze>(
            cache_pos_unsq,
            ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1}));
        q_unsq1->set_friendly_name("model.decoder.q_unsq1");
        auto q_unsq2 = std::make_shared<ov::opset11::Unsqueeze>(
            q_unsq1,
            ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {3}));
        q_unsq2->set_friendly_name("model.decoder.q_unsq2");
        // q_unsq2 shape: [1, 1, seq, 1]

        // Causal mask: kv_pos <= q_pos → [1, 1, seq, total_seq]
        auto causal_bool = std::make_shared<ov::op::v1::LessEqual>(kv_unsq3, q_unsq2);
        causal_bool->set_friendly_name("model.decoder.causal_mask_bool");

        // Broadcast to [batch, 1, seq, total_seq]
        auto batch_dim_b = std::make_shared<ov::opset11::Gather>(
            ids_shape,
            ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {0}),
            ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0}));
        auto seq_len_1d = std::make_shared<ov::opset11::Reshape>(
            seq_len, ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1}), false);
        auto total_seq_1d = std::make_shared<ov::opset11::Unsqueeze>(
            total_seq_len,
            ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0}));
        auto broadcast_shape = std::make_shared<ov::opset11::Concat>(
            ov::OutputVector{
                batch_dim_b->output(0),
                ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1})->output(0),
                seq_len_1d->output(0),
                total_seq_1d->output(0)},
            0);
        auto causal_broadcast = std::make_shared<ov::op::v3::Broadcast>(
            causal_bool, broadcast_shape, ov::op::BroadcastType::BIDIRECTIONAL);
        causal_broadcast->set_friendly_name("model.decoder.causal_mask_broadcast");

        // Select: bool → float (0.0 for visible, -inf for masked)
        // Always f32: NPUW's AttentionMask pattern matchers inject f32 nodes,
        // so the mask must be f32 regardless of model weight precision.
        auto select_true = ov::opset11::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});
        auto select_false = ov::opset11::Constant::create(ov::element::f32, ov::Shape{}, {-65504.0f});
        auto causal_float = std::make_shared<ov::op::v1::Select>(
            causal_broadcast, select_true, select_false);
        causal_float->set_friendly_name("model.decoder.causal_mask");

        // Slice on axis 3 from 0 to total_seq_len.
        // This is a structural no-op but NPUW's AttentionMaskInput (prefill)
        // looks for Slice → SDPA input[3] to identify self-attention nodes.
        auto slice_start = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto slice_stop = std::make_shared<ov::opset11::Reshape>(
            total_seq_len,
            ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1}), false);
        auto slice_step = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        auto slice_axes = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {3});
        auto causal_sliced = std::make_shared<ov::op::v8::Slice>(
            causal_float, slice_start, slice_stop, slice_step, slice_axes);
        causal_sliced->set_friendly_name("model.decoder.causal_mask_sliced");
        // Mask must be f32: NPUW's AttentionMaskInput matcher requires
        // Slice to be the immediate source of SDPA input[3], and its
        // injected replacement nodes are all f32.
        shared_mask = causal_sliced->output(0);
    }

    // Decoder layers
    LayerNorm norm(d, prec);
    GELUFn ffn(d, config.decoder_ffn_dim, prec, wf, true);  // bias=true

    ov::Output<ov::Node> current = hidden_states->output(0);
    ov::SinkVector all_sinks;

    for (size_t layer = 0; layer < config.decoder_layers; ++layer) {
        std::string prefix = "model.decoder.layers." + std::to_string(layer) + ".";

        // Self-attention (with KV cache, causal mask)
        Attention self_attn{d, heads, heads, hd, prec, wf};
        self_attn.add_bias = true;
        self_attn.cache_mode = Attention::CacheMode::ConcatBeam;
        self_attn.cache_infix = ".decoder.";
        self_attn.batch_source = input_ids->output(0);
        self_attn.beam_idx = beam_idx->output(0);
        self_attn.layer_idx = layer;
        self_attn.sdpa_mask = shared_mask;
        self_attn.o_proj_name = "self_attn.out_proj";
        if (layer == 0) {
            self_attn.prebuilt_k_variable = layer0_k_read.variable;
            self_attn.prebuilt_k_beam_gather = layer0_k_read.beam_gather;
        }

        // Cross-attention to encoder (uses converted encoder hidden states)
        Attention cross_attn{d, heads, heads, hd, prec, wf};
        cross_attn.add_bias = true;
        cross_attn.kv_source = enc_hs;
        cross_attn.cache_mode = Attention::CacheMode::StoreOnly;
        cross_attn.cache_infix = ".encoder.";
        cross_attn.layer_idx = layer;
        cross_attn.q_proj_name = "encoder_attn.q_proj";
        cross_attn.k_proj_name = "encoder_attn.k_proj";
        cross_attn.v_proj_name = "encoder_attn.v_proj";
        cross_attn.o_proj_name = "encoder_attn.out_proj";

        auto layer_result = make_whisper_decoder_layer(current, norm, self_attn, cross_attn, ffn, prefix);
        current = layer_result.output;

        for (auto& sink : layer_result.sinks) {
            all_sinks.push_back(std::dynamic_pointer_cast<ov::op::Sink>(sink));
        }
    }

    // Final LayerNorm
    auto final_norm = norm(current, "model.decoder.layer_norm");

    // LM head
    auto logits = make_lm_head(final_norm, d, config.vocab_size, "proj_out", prec, wf);

    // Result — always f32 output (WhisperPipeline reads logits as f32)
    ov::Output<ov::Node> logits_out = logits;
    if (prec != ov::element::f32) {
        auto cvt = std::make_shared<ov::op::v0::Convert>(logits, ov::element::f32);
        cvt->set_friendly_name("model.decoder.logits_convert");
        logits_out = cvt->output(0);
    }
    result(logits_out, "logits");

    return std::make_shared<ov::Model>(m_results, all_sinks, m_parameters, "whisper_decoder");
}

// ============================================================================
// BERT Encoder Builder
// ============================================================================

std::shared_ptr<ov::Model> ModelBuilder::build_bert_encoder(const BERTConfig& config_in) {
    clear();

    BERTConfig config = config_in;
    if (!config.weight)
        config.weight = FP32Weight{};
    if (!config.norm)
        config.norm = LayerNorm(config.hidden_size, config.precision);
    if (!config.ffn)
        config.ffn = GELUFn(config.hidden_size, config.intermediate_size, config.precision, config.weight, true);

    const auto prec = config.precision;
    const auto hs = config.hidden_size;

    // ===== Inputs =====
    auto input_ids = parameter(ov::element::i64, ov::PartialShape{-1, -1}, "input_ids");
    auto attention_mask = parameter(ov::element::i64, ov::PartialShape{-1, -1}, "attention_mask");
    auto token_type_ids = parameter(ov::element::i64, ov::PartialShape{-1, -1}, "token_type_ids");

    // ===== Embeddings: word + position + token_type =====
    auto word_embed = make_embedding(input_ids->output(0), config.vocab_size, hs,
                                     "embeddings.word_embeddings", prec);

    // Position embeddings: arange(0, seq_len) -> Gather from learned table
    auto ids_shape = std::make_shared<ov::opset11::ShapeOf>(input_ids, ov::element::i64);
    auto idx1 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto axis0 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto seq_len = std::make_shared<ov::opset11::Gather>(ids_shape, idx1, axis0);
    auto start = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto step = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto pos_ids = std::make_shared<ov::op::v4::Range>(start, seq_len, step, ov::element::i64);
    pos_ids->set_friendly_name("embeddings.position_ids");
    auto pos_embed = make_embedding(pos_ids->output(0), config.max_position_embeddings, hs,
                                    "embeddings.position_embeddings", prec);

    // Token type embeddings
    auto type_embed = make_embedding(token_type_ids->output(0), config.type_vocab_size, hs,
                                     "embeddings.token_type_embeddings", prec);

    // Sum embeddings + LayerNorm
    auto embed_sum1 = std::make_shared<ov::opset11::Add>(word_embed, pos_embed);
    embed_sum1->set_friendly_name("embeddings.add_pos");
    auto embed_sum2 = std::make_shared<ov::opset11::Add>(embed_sum1, type_embed);
    embed_sum2->set_friendly_name("embeddings.add_type");
    auto embed_normed = config.norm(embed_sum2->output(0), "embeddings.LayerNorm");

    // ===== Attention mask (padding only, no causal) =====
    auto sdpa_mask = make_padding_mask(attention_mask->output(0), prec);

    // ===== Encoder Layers (post-norm) =====
    ov::Output<ov::Node> current = embed_normed;

    for (size_t layer = 0; layer < config.num_layers; ++layer) {
        std::string prefix = "encoder.layer." + std::to_string(layer) + ".";
        Attention attn{hs, config.num_heads, config.num_heads, config.head_dim, prec, config.weight};
        attn.add_bias = true;
        attn.sdpa_mask = sdpa_mask;
        attn.layer_idx = layer;
        auto layer_result = make_post_norm_layer(current, config.norm, attn, config.ffn, prefix);
        current = layer_result.output;
    }

    // ===== Output =====
    result(current, "last_hidden_state");

    return std::make_shared<ov::Model>(m_results, ov::SinkVector{}, m_parameters, "bert_encoder");
}

}  // namespace npuw
}  // namespace test
}  // namespace ov
