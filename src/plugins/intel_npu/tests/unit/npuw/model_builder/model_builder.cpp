// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "model_builder.hpp"

#include <algorithm>
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

// Named constants for magic values used throughout model construction.
constexpr float kRoPEBaseFrequency = 10000.0f;
constexpr float kAttentionMaskPadding = -10000.0f;
constexpr float kAttentionMaskPaddingFP16Min = -65504.0f;

static float fill_value_from_name(const std::string& name) {
    size_t h = std::hash<std::string>{}(name);
    return 0.01f + static_cast<float>(h % 100000u) / 100000.0f;  // [0.01, 1.01)
}

static int8_t int_fill_from_name(const std::string& name, ov::element::Type type) {
    size_t h = std::hash<std::string>{}(name);
    if (type == ov::element::i4)
        return static_cast<int8_t>(1 + (h % 6));
    if (type == ov::element::u4)
        return static_cast<int8_t>(1 + (h % 14));
    return static_cast<int8_t>(1 + (h % 100));
}

ov::Output<ov::Node> FloatWeight::operator()(const std::string& name,
                                             const ov::Shape& shape,
                                             ov::element::Type compute_precision) const {
    // Use unique fill values per constant to prevent CSE from merging
    // different projections (e.g. Q/K/V) that happen to share dimensions.
    float fill_val = fill_value_from_name(name);
    auto weight =
        ov::opset11::Constant::create(storage_type, shape, std::vector<float>(ov::shape_size(shape), fill_val));
    weight->set_friendly_name(name);

    if (storage_type == compute_precision) {
        return weight->output(0);
    }
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
    int8_t fill_val = int_fill_from_name(name, storage_type);
    auto weight = ov::opset11::Constant::create(storage_type, shape, std::vector<int8_t>(rows * cols, fill_val));
    weight->set_friendly_name(name);

    auto convert = std::make_shared<ov::opset11::Convert>(weight, ov::element::f16);
    convert->set_friendly_name(name + "_convert");

    ov::Output<ov::Node> decompressed;

    if (group_size > 0) {
        // Group quantization: reshape -> per-group scale -> reshape back
        OPENVINO_ASSERT(cols >= group_size && cols % group_size == 0,
                        "Group quantization requires cols (",
                        cols,
                        ") >= group_size (",
                        group_size,
                        ") and evenly divisible");
        const size_t num_groups = cols / group_size;

        auto reshape_shape = ov::opset11::Constant::create(ov::element::i64,
                                                           ov::Shape{3},
                                                           std::vector<int64_t>{static_cast<int64_t>(rows),
                                                                                static_cast<int64_t>(num_groups),
                                                                                static_cast<int64_t>(group_size)});

        auto reshaped = std::make_shared<ov::opset11::Reshape>(convert, reshape_shape, false);
        reshaped->set_friendly_name(name + "_group_reshape");

        float scale_val = fill_value_from_name(name + "_scale");
        auto scale = ov::opset11::Constant::create(ov::element::f16,
                                                   ov::Shape{rows, num_groups, 1},
                                                   std::vector<float>(rows * num_groups, scale_val));
        scale->set_friendly_name(name + "_scale");

        auto scaled = std::make_shared<ov::opset11::Multiply>(reshaped, scale);
        scaled->set_friendly_name(name + "_decompress");

        auto out_shape =
            ov::opset11::Constant::create(ov::element::i64,
                                          ov::Shape{2},
                                          std::vector<int64_t>{static_cast<int64_t>(rows), static_cast<int64_t>(cols)});

        auto back = std::make_shared<ov::opset11::Reshape>(scaled, out_shape, false);
        back->set_friendly_name(name + "_group_reshape_back");

        decompressed = back->output(0);
    } else {
        // Per-channel scale
        float scale_val = fill_value_from_name(name + "_scale");
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

ov::Output<ov::Node> LayerNorm::operator()(const ov::Output<ov::Node>& input, const std::string& name) const {
    float w_val = 1.0f + fill_value_from_name(name + ".weight") * 0.1f;
    float b_val = fill_value_from_name(name + ".bias") * 0.01f;
    auto weight =
        ov::opset11::Constant::create(precision, ov::Shape{hidden_size}, std::vector<float>(hidden_size, w_val));
    weight->set_friendly_name(name + ".weight");

    auto bias =
        ov::opset11::Constant::create(precision, ov::Shape{hidden_size}, std::vector<float>(hidden_size, b_val));
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
    float w_val = 1.0f + fill_value_from_name(name + ".weight") * 0.1f;
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

/// Builds frequency cos/sin chain matching NPUW's AddPositionIdsNode pattern.
struct RoPEFrequencies {
    ov::Output<ov::Node> cos, sin;
};

static RoPEFrequencies build_rope_frequencies(size_t head_dim,
                                              ov::element::Type precision,
                                              const ov::Output<ov::Node>& position_ids,
                                              const std::string& prefix = "model.rope") {
    const size_t half_dim = head_dim / 2;

    std::vector<float> inv_freq_data(half_dim);
    for (size_t i = 0; i < half_dim; ++i) {
        inv_freq_data[i] = 1.0f / std::pow(kRoPEBaseFrequency, static_cast<float>(2 * i) / static_cast<float>(head_dim));
    }
    auto inv_freq = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1, half_dim, 1}, inv_freq_data);
    inv_freq->set_friendly_name(prefix + ".inv_freq");

    // position_ids [batch, seq] -> Unsqueeze -> Convert(f32) -> MatMul(inv_freq)
    // -> Transpose -> Concat(self,self) -> Sin/Cos -> Unsqueeze [batch, 1, seq, head_dim]
    auto unsq_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto unsqueezed = std::make_shared<ov::opset11::Unsqueeze>(position_ids, unsq_axis);
    unsqueezed->set_friendly_name(prefix + ".pos_unsqueeze");

    auto converted = std::make_shared<ov::opset11::Convert>(unsqueezed, ov::element::f32);
    converted->set_friendly_name(prefix + ".pos_convert");

    auto matmul = std::make_shared<ov::opset11::MatMul>(inv_freq, converted, false, false);
    matmul->set_friendly_name(prefix + ".freq_matmul");

    auto perm = ov::opset11::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 2, 1});
    auto transposed = std::make_shared<ov::opset11::Transpose>(matmul, perm);
    transposed->set_friendly_name(prefix + ".freq_transpose");

    auto concat =
        std::make_shared<ov::opset11::Concat>(ov::OutputVector{transposed->output(0), transposed->output(0)}, -1);
    concat->set_friendly_name(prefix + ".freq_concat");

    auto sin_node = std::make_shared<ov::op::v0::Sin>(concat);
    sin_node->set_friendly_name(prefix + ".sin");
    auto cos_node = std::make_shared<ov::op::v0::Cos>(concat);
    cos_node->set_friendly_name(prefix + ".cos");

    auto head_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1});

    auto cos_unsq = std::make_shared<ov::opset11::Unsqueeze>(cos_node, head_axis);
    cos_unsq->set_friendly_name(prefix + ".cos_unsqueeze");

    auto sin_unsq = std::make_shared<ov::opset11::Unsqueeze>(sin_node, head_axis);
    sin_unsq->set_friendly_name(prefix + ".sin_unsqueeze");

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

HalfRotationRoPE::HalfRotationRoPE(size_t hd, ov::element::Type precision, const ov::Output<ov::Node>& position_ids)
    : head_dim(hd) {
    auto freq = build_rope_frequencies(hd, precision, position_ids);
    cos_freq = freq.cos;
    sin_freq = freq.sin;
}

ov::Output<ov::Node> HalfRotationRoPE::operator()(const ov::Output<ov::Node>& input, const std::string& name) const {
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

    auto neg_one = ov::opset11::Constant::create(input.get_element_type(), ov::Shape{}, {-1.0f});
    auto neg_second = std::make_shared<ov::opset11::Multiply>(second_half, neg_one);
    neg_second->set_friendly_name(name + "_neg_second");

    auto rotated =
        std::make_shared<ov::opset11::Concat>(ov::OutputVector{neg_second->output(0), first_half->output(0)}, -1);
    rotated->set_friendly_name(name + "_rotated");

    auto input_cos = std::make_shared<ov::opset11::Multiply>(input, cos_freq);
    input_cos->set_friendly_name(name + "_input_cos");

    auto rotated_sin = std::make_shared<ov::opset11::Multiply>(rotated, sin_freq);
    rotated_sin->set_friendly_name(name + "_rotated_sin");

    auto output = std::make_shared<ov::opset11::Add>(input_cos, rotated_sin);
    output->set_friendly_name(name);

    return output->output(0);
}

InterleavedRoPE::InterleavedRoPE(size_t hd, ov::element::Type precision, const ov::Output<ov::Node>& position_ids)
    : head_dim(hd) {
    auto freq = build_rope_frequencies(hd, precision, position_ids);
    cos_freq = freq.cos;
    sin_freq = freq.sin;
}

ov::Output<ov::Node> InterleavedRoPE::operator()(const ov::Output<ov::Node>& input, const std::string& name) const {
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

ov::Output<ov::Node> make_position_ids_2d() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
    param->set_friendly_name("position_ids");
    param->output(0).set_names({"position_ids"});
    return param->output(0);
}

ov::Output<ov::Node> make_position_ids_3d() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{3, -1, -1});
    param->set_friendly_name("position_ids");
    param->output(0).set_names({"position_ids"});

    auto indices = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto gather = std::make_shared<ov::opset11::Gather>(param, indices, axis);

    auto squeeze_axes = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto squeeze = std::make_shared<ov::opset11::Squeeze>(gather, squeeze_axes);
    squeeze->set_friendly_name("position_ids_2d");

    return squeeze->output(0);
}

ov::Output<ov::Node> make_linear(const ov::Output<ov::Node>& input,
                                 size_t in_features,
                                 size_t out_features,
                                 const std::string& name,
                                 ov::element::Type precision,
                                 const WeightFn& weight_fn,
                                 const WeightFn& bias_fn) {
    auto weight_output = weight_fn(name + ".weight", ov::Shape{out_features, in_features}, precision);

    auto matmul = std::make_shared<ov::opset11::MatMul>(input, weight_output, false, true);
    matmul->set_friendly_name(name);

    if (bias_fn) {
        auto bias = bias_fn(name + ".bias", ov::Shape{out_features}, precision);
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

    if (!shared_broadcast_shape.get_node() && n_rep == 1) {
        return kv;
    }

    OPENVINO_ASSERT(num_heads % actual_kv_heads == 0,
                    "num_heads (",
                    num_heads,
                    ") must be divisible by num_kv_heads (",
                    actual_kv_heads,
                    ")");

    ov::Output<ov::Node> broadcast_shape_output;
    if (shared_broadcast_shape.get_node()) {
        broadcast_shape_output = shared_broadcast_shape;
    } else {
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

    auto concat = std::make_shared<ov::opset11::Concat>(ov::OutputVector{read_state.beam_gather, current_kv}, 2);
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
                                           const WeightFn& weight_fn,
                                           const WeightFn& bias_fn) {
    auto attn_trans = make_attention_transpose(sdpa_output, name + "_transpose");

    auto reshape_shape = ov::opset11::Constant::create(ov::element::i64,
                                                       ov::Shape{3},
                                                       std::vector<int64_t>{0, -1, static_cast<int64_t>(hidden_size)});
    auto attn_reshaped = std::make_shared<ov::opset11::Reshape>(attn_trans, reshape_shape, true);
    attn_reshaped->set_friendly_name(name + "_reshape");

    return make_linear(attn_reshaped->output(0), hidden_size, hidden_size, name, precision, weight_fn, bias_fn);
}

ov::Output<ov::Node> make_embedding(const ov::Output<ov::Node>& input_ids,
                                    size_t vocab_size,
                                    size_t hidden_size,
                                    const std::string& name,
                                    ov::element::Type precision) {
    float fill_val = fill_value_from_name(name);
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
    return make_linear(hidden_states, hidden_size, vocab_size, name, precision, weight_fn);
}

ov::Output<ov::Node> make_conv1d(const ov::Output<ov::Node>& input,
                                 size_t in_channels,
                                 size_t out_channels,
                                 size_t kernel_size,
                                 size_t stride,
                                 size_t padding,
                                 const std::string& name,
                                 ov::element::Type precision) {
    float w_val = fill_value_from_name(name + ".weight");
    float b_val = fill_value_from_name(name + ".bias") * 0.01f;
    auto weight = ov::opset11::Constant::create(precision,
                                                ov::Shape{out_channels, in_channels, kernel_size},
                                                std::vector<float>(out_channels * in_channels * kernel_size, w_val));
    weight->set_friendly_name(name + ".weight");

    auto conv = std::make_shared<ov::op::v1::Convolution>(input,
                                                          weight,
                                                          ov::Strides{stride},
                                                          ov::CoordinateDiff{static_cast<std::ptrdiff_t>(padding)},
                                                          ov::CoordinateDiff{static_cast<std::ptrdiff_t>(padding)},
                                                          ov::Strides{1});
    conv->set_friendly_name(name);

    auto bias = ov::opset11::Constant::create(precision,
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

    // No Gather/beam reorder — encoder KV is identical across beams
    auto read_value = std::make_shared<ov::op::v6::ReadValue>(encoder_kv, variable);
    read_value->set_friendly_name(name + "_read");

    auto assign = std::make_shared<ov::op::v6::Assign>(read_value, variable);
    assign->set_friendly_name(name + "_assign");

    return {read_value->output(0), {}, assign};
}

ov::Output<ov::Node> SwiGLU::operator()(const ov::Output<ov::Node>& input, const std::string& name) const {
    auto gate = make_linear(input, hidden_size, intermediate_size, name + ".gate_proj", precision, weight_fn);
    auto up = make_linear(input, hidden_size, intermediate_size, name + ".up_proj", precision, weight_fn);

    auto sigmoid = std::make_shared<ov::opset11::Sigmoid>(gate);

    auto silu = std::make_shared<ov::opset11::Multiply>(gate, sigmoid);
    silu->set_friendly_name(name + "_silu");

    auto gate_up = std::make_shared<ov::opset11::Multiply>(silu, up);
    gate_up->set_friendly_name(name + "_gate_up");

    auto down = make_linear(gate_up, intermediate_size, hidden_size, name + ".down_proj", precision, weight_fn);

    return down;
}

ov::Output<ov::Node> GELU::operator()(const ov::Output<ov::Node>& input, const std::string& name) const {
    auto up = make_linear(input, hidden_size, intermediate_size, name + ".up_proj", precision, weight_fn, bias_fn);

    auto gelu = std::make_shared<ov::opset11::Gelu>(up);
    gelu->set_friendly_name(name + "_gelu");

    auto down = make_linear(gelu, intermediate_size, hidden_size, name + ".down_proj", precision, weight_fn, bias_fn);

    return down;
}

ov::Output<ov::Node> make_transformer_layers(const ov::Output<ov::Node>& initial,
                                             size_t num_layers,
                                             const std::string& prefix_base,
                                             const LayerFn& layer_fn) {
    ov::Output<ov::Node> current = initial;
    for (size_t layer = 0; layer < num_layers; ++layer) {
        std::string prefix = prefix_base + std::to_string(layer) + ".";
        current = layer_fn(current, prefix, layer);
    }
    return current;
}

/// Shared GQA broadcast shape — ReConstructEmbeddingModel requires pointer equality across SDPAs.
static ov::Output<ov::Node> make_shared_gqa_broadcast(const ov::Output<ov::Node>& shape_source,
                                                      size_t kv_heads,
                                                      size_t num_heads,
                                                      size_t head_dim) {
    const size_t n_rep = num_heads / kv_heads;
    auto shape_of = std::make_shared<ov::opset11::ShapeOf>(shape_source, ov::element::i64);
    auto axis0 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto idx0 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto batch_dim = std::make_shared<ov::opset11::Gather>(shape_of, idx0, axis0);
    auto idx1 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto seq_dim = std::make_shared<ov::opset11::Gather>(shape_of, idx1, axis0);
    auto kv_heads_const =
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(kv_heads)});
    auto n_rep_const = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(n_rep)});
    auto head_dim_const =
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(head_dim)});
    auto shared_concat = std::make_shared<ov::opset11::Concat>(
        ov::OutputVector{batch_dim, kv_heads_const, n_rep_const, seq_dim, head_dim_const},
        0);
    shared_concat->set_friendly_name("model.shared_gqa_broadcast_shape");
    return shared_concat->output(0);
}

/// Missing separator (e.g. "keypresent") is intentional — matches OV's StatefulToStateless pass.
static std::string make_kv_var_id(const std::string& layer, const std::string& infix, const std::string& kv_type) {
    return "past_key_values." + layer + infix + kv_type + "present." + layer + infix + kv_type;
}

ov::Output<ov::Node> Attention::operator()(const ov::Output<ov::Node>& q,
                                           const ov::Output<ov::Node>& k,
                                           const ov::Output<ov::Node>& v,
                                           const std::string& prefix,
                                           size_t layer_idx) const {
    auto q_reshaped = make_multihead_reshape(q, num_heads, head_dim, prefix + attn_prefix + "q_reshape");
    auto k_reshaped = make_multihead_reshape(k, num_kv_heads, head_dim, prefix + attn_prefix + "k_reshape");
    auto v_reshaped = make_multihead_reshape(v, num_kv_heads, head_dim, prefix + attn_prefix + "v_reshape");

    ov::Output<ov::Node> q_normed = q_reshaped;
    ov::Output<ov::Node> k_normed = k_reshaped;
    if (qk_norm) {
        q_normed = qk_norm(q_reshaped, prefix + attn_prefix + "q_norm");
        k_normed = qk_norm(k_reshaped, prefix + attn_prefix + "k_norm");
    }

    // [batch, seq, heads, dim] -> [batch, heads, seq, dim]
    auto q_trans = make_attention_transpose(q_normed, prefix + attn_prefix + "q_transpose");
    auto k_trans = make_attention_transpose(k_normed, prefix + attn_prefix + "k_transpose");
    auto v_trans = make_attention_transpose(v_reshaped, prefix + attn_prefix + "v_transpose");

    ov::Output<ov::Node> q_roped = q_trans;
    ov::Output<ov::Node> k_roped = k_trans;
    if (rope_fn) {
        q_roped = rope_fn(q_trans, prefix + "q_rope");
        k_roped = rope_fn(k_trans, prefix + "k_rope");
    }

    ov::Output<ov::Node> k_for_attn = k_roped;
    ov::Output<ov::Node> v_for_attn = v_trans;
    if (kv_cache_fn) {
        std::tie(k_for_attn, v_for_attn) = kv_cache_fn(k_roped, v_trans, layer_idx);
    }

    auto k_expanded =
        make_repeat_kv(k_for_attn, num_heads, num_kv_heads, head_dim, prefix + "k_repeat", shared_broadcast_shape);
    auto v_expanded =
        make_repeat_kv(v_for_attn, num_heads, num_kv_heads, head_dim, prefix + "v_repeat", shared_broadcast_shape);

    // 5-input SDPA with explicit scale needed for embedding model pattern matching
    size_t sdpa_scale_dim = shared_broadcast_shape.get_node() ? head_dim : 0;
    auto attn_output =
        make_sdpa(q_roped, k_expanded, v_expanded, prefix + attn_prefix + "attn", sdpa_mask, sdpa_scale_dim);

    return make_attention_output(attn_output, hidden_size, prefix + o_proj_name, precision, weight_fn, bias_fn);
}

ov::Output<ov::Node> Attention::operator()(const ov::Output<ov::Node>& input,
                                           const ov::Output<ov::Node>& kv_input,
                                           const std::string& prefix,
                                           size_t layer_idx) const {
    auto kv_src = kv_input.get_node() ? kv_input : input;
    size_t kv_dim = num_kv_heads * head_dim;
    auto q =
        make_linear(input, hidden_size, hidden_size, prefix + attn_prefix + "q_proj", precision, weight_fn, bias_fn);
    auto k = make_linear(kv_src, hidden_size, kv_dim, prefix + attn_prefix + "k_proj", precision, weight_fn, bias_fn);
    auto v = make_linear(kv_src, hidden_size, kv_dim, prefix + attn_prefix + "v_proj", precision, weight_fn, bias_fn);
    return (*this)(q, k, v, prefix, layer_idx);
}

/// Padding-only mask: [batch, seq] -> [batch, 1, 1, seq] float (0.0=attend, -10000.0=pad)
static ov::Output<ov::Node> make_padding_mask(const ov::Output<ov::Node>& attention_mask_output,
                                              ov::element::Type prec) {
    auto mask_float = std::make_shared<ov::opset11::Convert>(attention_mask_output, prec);
    mask_float->set_friendly_name("model.mask_convert");

    auto one_const = ov::opset11::Constant::create(prec, ov::Shape{}, {1.0f});
    auto inv_mask = std::make_shared<ov::opset11::Subtract>(one_const, mask_float);
    inv_mask->set_friendly_name("model.mask_invert");

    auto neg_inf = ov::opset11::Constant::create(prec, ov::Shape{}, {kAttentionMaskPadding});
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
    auto padding_4d = make_padding_mask(attention_mask_output, prec);

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
    auto select_false = ov::opset11::Constant::create(prec, ov::Shape{}, {kAttentionMaskPadding});

    auto causal_float = std::make_shared<ov::op::v1::Select>(causal_bool, select_true, select_false);
    causal_float->set_friendly_name("model.causal_mask");

    auto unsqueeze_axes = ov::opset11::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});

    auto causal_4d = std::make_shared<ov::opset11::Unsqueeze>(causal_float, unsqueeze_axes);
    causal_4d->set_friendly_name("model.causal_mask_4d");

    auto combined = std::make_shared<ov::opset11::Add>(padding_4d, causal_4d);
    combined->set_friendly_name("model.mask_4d");

    return combined->output(0);
}

std::shared_ptr<ov::Model> ModelBuilder::get_model_with_one_op() {
    auto param = std::make_shared<ov::opset11::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
    param->set_friendly_name("input");
    auto const_value = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto add = std::make_shared<ov::opset11::Add>(param, const_value);
    add->set_friendly_name("add");
    auto result = std::make_shared<ov::opset11::Result>(add);
    result->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::OutputVector{result->output(0)});
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

    return std::make_shared<ov::Model>(ov::OutputVector{result->output(0)});
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
    ov::OutputVector outputs;

    // Add Results for specified blocks
    for (size_t idx : block_indices) {
        if (idx < block_outputs.size()) {
            auto result = std::make_shared<ov::op::v0::Result>(block_outputs[idx]);
            m_nodes.push_back(result);
            set_name(result);
            outputs.push_back(result->output(0));
        }
    }

    // Always add final tail Result
    auto final_result = std::make_shared<ov::op::v0::Result>(tail[5]);
    m_nodes.push_back(final_result);
    set_name(final_result);
    outputs.push_back(final_result->output(0));

    return std::make_shared<ov::Model>(outputs);
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

    return std::make_shared<ov::Model>(ov::OutputVector{result->output(0)});
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

    ov::OutputVector outputs;
    auto tail_result = std::make_shared<ov::opset11::Result>(tail_add);
    m_nodes.push_back(tail_result);
    set_name(tail_result);
    outputs.push_back(tail_result->output(0));

    if (last_block_has_direct_result) {
        auto direct_result = std::make_shared<ov::opset11::Result>(current_values);
        m_nodes.push_back(direct_result);
        set_name(direct_result);
        outputs.push_back(direct_result->output(0));
    }

    return std::make_shared<ov::Model>(outputs);
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

std::shared_ptr<ov::op::v0::Parameter> ModelBuilder::parameter(ov::element::Type type,
                                                               const ov::PartialShape& shape,
                                                               const std::string& name) {
    auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
    param->set_friendly_name(name);
    param->output(0).set_names({name});
    return param;
}

void ModelBuilder::clear() {
    m_nodes.clear();
    m_sinks.clear();
    m_name_idx = 0;
}

ov::Output<ov::Node> ModelBuilder::setup_position_ids(ModelConfig& config, const ov::Output<ov::Node>& seq_source) {
    OPENVINO_ASSERT(!(config.internal_position_ids && config.position_ids.get_node()),
                    "internal_position_ids and position_ids are mutually exclusive");
    ov::Output<ov::Node> position_ids_output;

    if (config.internal_position_ids) {
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
        // RoPE chain (Range → Unsqueeze → ... → Sin/Cos) matches NPUW's AddPositionIdsNode pattern
    } else if (config.position_ids.get_node()) {
        position_ids_output = config.position_ids;
    } else if (!config.rope) {
        position_ids_output = make_position_ids_2d();
    }
    // config.rope set without position_ids means RoPE was pre-built with position_ids baked in
    if (position_ids_output.get_node() && !config.rope) {
        config.rope = HalfRotationRoPE(config.head_dim, config.precision, position_ids_output);
    }

    return position_ids_output;
}

std::shared_ptr<ov::Model> ModelBuilder::make_model(const ov::Output<ov::Node>& output,
                                                    const std::string& result_name,
                                                    const std::string& model_name) {
    auto res = std::make_shared<ov::op::v0::Result>(output);
    res->set_friendly_name(result_name);
    res->output(0).set_names({result_name});

    return std::make_shared<ov::Model>(ov::OutputVector{res->output(0)}, m_sinks, model_name);
}

std::shared_ptr<ov::Model> ModelBuilder::build_model(const ModelConfig& config) {
    OPENVINO_ASSERT(
        (int)config.use_conv_features + (int)config.use_cross_attention + (int)config.use_token_type_embedding <= 1,
        "At most one structural dispatch flag may be set");
    if (config.use_conv_features) {
        return build_whisper_encoder(config);
    }
    if (config.use_cross_attention) {
        return build_whisper_decoder(config);
    }
    if (config.use_token_type_embedding) {
        return build_embedding_encoder(config);
    }
    return build_llm(config);
}

std::shared_ptr<ov::Model> ModelBuilder::build_llm(const ModelConfig& config_in) {
    clear();

    ModelConfig config = config_in;
    const auto prec = config.precision;

    auto attention_mask = parameter(ov::element::i64, ov::PartialShape{-1, -1}, "attention_mask");

    ov::Output<ov::Node> hidden_states;
    ov::Output<ov::Node> seq_source;

    if (config.use_inputs_embeds) {
        auto inputs_embeds =
            parameter(prec, ov::PartialShape{-1, -1, static_cast<int64_t>(config.hidden_size)}, "inputs_embeds");
        hidden_states = inputs_embeds->output(0);
        seq_source = inputs_embeds->output(0);
    } else {
        auto input_ids = parameter(ov::element::i64, ov::PartialShape{-1, -1}, "input_ids");
        hidden_states =
            make_embedding(input_ids->output(0), config.vocab_size, config.hidden_size, "model.embed_tokens", prec);
        seq_source = input_ids->output(0);
    }

    setup_position_ids(config, seq_source);

    ov::Output<ov::Node> beam_idx_output;
    if (config.use_kv_cache) {
        auto beam_idx = parameter(ov::element::i32, ov::PartialShape{-1}, "beam_idx");
        beam_idx_output = beam_idx->output(0);
    }

    auto sdpa_mask = make_causal_mask(seq_source, attention_mask->output(0), prec);

    // Shared GQA broadcast shape (embedding models only)
    ov::Output<ov::Node> shared_broadcast;
    if (!config.use_kv_cache && !config.lm_head_weight) {
        shared_broadcast = make_shared_gqa_broadcast(attention_mask->output(0),
                                                     config.get_kv_heads(),
                                                     config.num_heads,
                                                     config.head_dim);
    }

    const auto hs = config.hidden_size;
    const auto kv_heads = config.get_kv_heads();

    Attention attn{};
    attn.hidden_size = hs;
    attn.num_heads = config.num_heads;
    attn.num_kv_heads = kv_heads;
    attn.head_dim = config.head_dim;
    attn.precision = prec;
    attn.weight_fn = config.weight;
    attn.bias_fn = config.attn_bias;
    attn.qk_norm = config.qk_norm;
    attn.rope_fn = config.rope;
    attn.sdpa_mask = sdpa_mask;
    attn.shared_broadcast_shape = shared_broadcast;

    if (config.use_kv_cache) {
        attn.kv_cache_fn = [&](const ov::Output<ov::Node>& k,
                               const ov::Output<ov::Node>& v,
                               size_t layer) -> std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> {
            auto layer_str = std::to_string(layer);
            auto k_cache = make_kv_cache_concat(k,
                                                seq_source,
                                                beam_idx_output,
                                                kv_heads,
                                                config.head_dim,
                                                make_kv_var_id(layer_str, ".", "key"),
                                                prec);
            auto v_cache = make_kv_cache_concat(v,
                                                seq_source,
                                                beam_idx_output,
                                                kv_heads,
                                                config.head_dim,
                                                make_kv_var_id(layer_str, ".", "value"),
                                                prec);
            m_sinks.push_back(std::dynamic_pointer_cast<ov::op::Sink>(k_cache.assign));
            m_sinks.push_back(std::dynamic_pointer_cast<ov::op::Sink>(v_cache.assign));
            return {k_cache.concatenated, v_cache.concatenated};
        };
    }

    auto current =
        make_transformer_layers(hidden_states,
                                config.num_layers,
                                "model.layers.",
                                [&](const ov::Output<ov::Node>& input, const std::string& prefix, size_t layer) {
                                    if (config.pre_norm) {
                                        return make_pre_norm_layer(
                                            input,
                                            config.norm,
                                            [&](const ov::Output<ov::Node>& normed, const std::string& pfx) {
                                                return attn(normed, {}, pfx, layer);
                                            },
                                            config.ffn,
                                            prefix);
                                    } else {
                                        return make_post_norm_layer(
                                            input,
                                            config.norm,
                                            [&](const ov::Output<ov::Node>& inp, const std::string& pfx) {
                                                return attn(inp, {}, pfx, layer);
                                            },
                                            config.ffn,
                                            prefix);
                                    }
                                });

    auto final_norm = config.norm(current, "model.norm");

    std::string model_name = "synthetic_decoder";

    if (config.lm_head_weight) {
        auto logits =
            make_lm_head(final_norm, config.hidden_size, config.vocab_size, "lm_head", prec, config.lm_head_weight);
        return make_model(logits, "logits", model_name);
    }
    return make_model(final_norm, "last_hidden_state", model_name);
}

std::shared_ptr<ov::Model> ModelBuilder::build_whisper_encoder(const ModelConfig& config) {
    clear();
    const auto prec = config.precision;
    const auto d = config.hidden_size;

    auto input_features = parameter(ov::element::f32,
                                    ov::PartialShape{-1, static_cast<int64_t>(config.num_mel_bins),
                                                     static_cast<int64_t>(2 * config.max_source_positions)},
                                    "input_features");

    ov::Output<ov::Node> encoder_input = input_features->output(0);
    if (prec != ov::element::f32) {
        auto cvt = std::make_shared<ov::op::v0::Convert>(encoder_input, prec);
        cvt->set_friendly_name("model.encoder.input_convert");
        encoder_input = cvt->output(0);
    }

    auto conv1 = make_conv1d(encoder_input, config.num_mel_bins, d, 3, 1, 1, "model.encoder.conv1", prec);
    auto gelu1 = std::make_shared<ov::opset11::Gelu>(conv1);
    gelu1->set_friendly_name("model.encoder.conv1_gelu");

    auto conv2 = make_conv1d(gelu1->output(0), d, d, 3, 2, 1, "model.encoder.conv2", prec);
    auto gelu2 = std::make_shared<ov::opset11::Gelu>(conv2);
    gelu2->set_friendly_name("model.encoder.conv2_gelu");

    auto transpose_order = ov::opset11::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
    auto transposed = std::make_shared<ov::opset11::Transpose>(gelu2, transpose_order);
    transposed->set_friendly_name("model.encoder.transpose");

    auto pos_embed_val = fill_value_from_name("model.encoder.embed_positions.weight");
    auto pos_embed = ov::opset11::Constant::create(prec,
                                                   ov::Shape{1, config.max_source_positions, d},
                                                   std::vector<float>(config.max_source_positions * d, pos_embed_val));
    pos_embed->set_friendly_name("model.encoder.embed_positions.weight");

    auto embedded = std::make_shared<ov::opset11::Add>(transposed, pos_embed);
    embedded->set_friendly_name("model.encoder.pos_embed_add");

    Attention enc_attn{};
    enc_attn.hidden_size = d;
    enc_attn.num_heads = config.num_heads;
    enc_attn.num_kv_heads = config.num_heads;
    enc_attn.head_dim = config.head_dim;
    enc_attn.precision = prec;
    enc_attn.weight_fn = config.weight;
    enc_attn.bias_fn = config.attn_bias;
    enc_attn.o_proj_name = "self_attn.out_proj";

    auto current =
        make_transformer_layers(embedded->output(0),
                                config.get_encoder_layers(),
                                "model.encoder.layers.",
                                [&](const ov::Output<ov::Node>& input, const std::string& prefix, size_t /*layer*/) {
                                    return make_pre_norm_layer(
                                        input,
                                        config.norm,
                                        [&](const ov::Output<ov::Node>& normed, const std::string& pfx) {
                                            return enc_attn(normed, {}, pfx);
                                        },
                                        config.ffn,
                                        prefix);
                                });

    auto final_norm = config.norm(current, "model.encoder.layer_norm");

    // Always f32 output — WhisperPipeline reads encoder output as f32
    ov::Output<ov::Node> encoder_output = final_norm;
    if (prec != ov::element::f32) {
        auto cvt = std::make_shared<ov::op::v0::Convert>(final_norm, ov::element::f32);
        cvt->set_friendly_name("model.encoder.output_convert");
        encoder_output = cvt->output(0);
    }

    return make_model(encoder_output, "last_hidden_state", "synthetic_whisper_encoder");
}

struct CachePositionResult {
    ov::Output<ov::Node> position_ids;  // [batch, seq]
    ov::Output<ov::Node> total_seq_len;
    ov::Output<ov::Node> seq_len;
    ov::Output<ov::Node> cache_pos_unsq;  // [1, seq] (needed for causal mask)
    ov::Output<ov::Node> ids_shape;       // ShapeOf(input_ids)
};

static CachePositionResult make_cache_position_ids(const ov::Output<ov::Node>& input_ids,
                                                   const ov::Output<ov::Node>& kv_cache_beam_gather,
                                                   const std::string& prefix) {
    auto ids_shape = std::make_shared<ov::opset11::ShapeOf>(input_ids, ov::element::i64);
    ids_shape->set_friendly_name(prefix + "ids_shape");
    auto seq_len =
        std::make_shared<ov::opset11::Gather>(ids_shape,
                                              ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1}),
                                              ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    seq_len->set_friendly_name(prefix + "seq_len");

    // kv_seq_len = ShapeOf(beam_gather)[2] — root of the CachePositionInput pattern
    auto kv_shape = std::make_shared<ov::opset11::ShapeOf>(kv_cache_beam_gather, ov::element::i64);
    kv_shape->set_friendly_name(prefix + "kv_shape");
    auto kv_seq_len =
        std::make_shared<ov::opset11::Gather>(kv_shape,
                                              ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {2}),
                                              ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    kv_seq_len->set_friendly_name(prefix + "kv_seq_len");

    // CachePositionInput pattern: Gather -> Add -> Range -> Unsqueeze -> Tile
    auto total_seq_len = std::make_shared<ov::opset11::Add>(kv_seq_len, seq_len);
    total_seq_len->set_friendly_name(prefix + "total_seq_len");

    auto cache_positions = std::make_shared<ov::op::v4::Range>(
        kv_seq_len->output(0),
        total_seq_len->output(0),
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1})->output(0),
        ov::element::i64);
    cache_positions->set_friendly_name(prefix + "cache_positions");

    auto cache_pos_unsq =
        std::make_shared<ov::opset11::Unsqueeze>(cache_positions,
                                                 ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    cache_pos_unsq->set_friendly_name(prefix + "cache_pos_unsq");

    auto batch_dim_for_tile =
        std::make_shared<ov::opset11::Gather>(ids_shape,
                                              ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {0}),
                                              ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    batch_dim_for_tile->set_friendly_name(prefix + "batch_for_tile");

    auto tile_repeats = std::make_shared<ov::opset11::Concat>(
        ov::OutputVector{batch_dim_for_tile->output(0),
                         ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1})->output(0)},
        0);
    tile_repeats->set_friendly_name(prefix + "tile_repeats");

    auto position_ids = std::make_shared<ov::op::v0::Tile>(cache_pos_unsq, tile_repeats);
    position_ids->set_friendly_name(prefix + "position_ids");

    return {position_ids->output(0),
            total_seq_len->output(0),
            seq_len->output(0),
            cache_pos_unsq->output(0),
            ids_shape->output(0)};
}

static ov::Output<ov::Node> make_whisper_causal_mask(const CachePositionResult& cache_pos, const std::string& prefix) {
    // kv_idx: Range -> 3x Unsqueeze -> [1, 1, 1, total_seq]
    auto mask_range = std::make_shared<ov::op::v4::Range>(
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0})->output(0),
        cache_pos.total_seq_len,
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1})->output(0),
        ov::element::i64);
    mask_range->set_friendly_name(prefix + "mask_range");

    auto kv_unsq1 =
        std::make_shared<ov::opset11::Unsqueeze>(mask_range,
                                                 ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    kv_unsq1->set_friendly_name(prefix + "kv_unsq1");
    auto kv_unsq2 =
        std::make_shared<ov::opset11::Unsqueeze>(kv_unsq1,
                                                 ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1}));
    kv_unsq2->set_friendly_name(prefix + "kv_unsq2");
    auto kv_unsq3 =
        std::make_shared<ov::opset11::Unsqueeze>(kv_unsq2,
                                                 ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {2}));
    kv_unsq3->set_friendly_name(prefix + "kv_unsq3");

    // q_idx: cache_pos_unsq -> 2x Unsqueeze -> [1, 1, seq, 1]
    auto q_unsq1 =
        std::make_shared<ov::opset11::Unsqueeze>(cache_pos.cache_pos_unsq,
                                                 ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1}));
    q_unsq1->set_friendly_name(prefix + "q_unsq1");
    auto q_unsq2 =
        std::make_shared<ov::opset11::Unsqueeze>(q_unsq1,
                                                 ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {3}));
    q_unsq2->set_friendly_name(prefix + "q_unsq2");

    auto causal_bool = std::make_shared<ov::op::v1::LessEqual>(kv_unsq3, q_unsq2);
    causal_bool->set_friendly_name(prefix + "causal_mask_bool");

    auto batch_dim_b =
        std::make_shared<ov::opset11::Gather>(cache_pos.ids_shape,
                                              ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {0}),
                                              ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    auto seq_len_1d =
        std::make_shared<ov::opset11::Reshape>(cache_pos.seq_len,
                                               ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1}),
                                               false);
    auto total_seq_1d =
        std::make_shared<ov::opset11::Unsqueeze>(cache_pos.total_seq_len,
                                                 ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    auto broadcast_shape = std::make_shared<ov::opset11::Concat>(
        ov::OutputVector{batch_dim_b->output(0),
                         ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1})->output(0),
                         seq_len_1d->output(0),
                         total_seq_1d->output(0)},
        0);
    auto causal_broadcast =
        std::make_shared<ov::op::v3::Broadcast>(causal_bool, broadcast_shape, ov::op::BroadcastType::BIDIRECTIONAL);
    causal_broadcast->set_friendly_name(prefix + "causal_mask_broadcast");

    // Always f32 — NPUW's AttentionMask matchers inject f32 nodes
    auto select_true = ov::opset11::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});
    auto select_false = ov::opset11::Constant::create(ov::element::f32, ov::Shape{}, {kAttentionMaskPaddingFP16Min});
    auto causal_float = std::make_shared<ov::op::v1::Select>(causal_broadcast, select_true, select_false);
    causal_float->set_friendly_name(prefix + "causal_mask");

    // Structural no-op Slice — AttentionMaskInput (prefill) needs Slice -> SDPA input[3]
    auto slice_start = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto slice_stop =
        std::make_shared<ov::opset11::Reshape>(cache_pos.total_seq_len,
                                               ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1}),
                                               false);
    auto slice_step = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto slice_axes = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {3});
    auto causal_sliced =
        std::make_shared<ov::op::v8::Slice>(causal_float, slice_start, slice_stop, slice_step, slice_axes);
    causal_sliced->set_friendly_name(prefix + "causal_mask_sliced");

    return causal_sliced->output(0);
}

static ov::Output<ov::Node> make_whisper_positional_embedding(const ov::Output<ov::Node>& token_embed,
                                                              const ov::Output<ov::Node>& position_ids,
                                                              size_t max_target_positions,
                                                              size_t hidden_size,
                                                              ov::element::Type precision,
                                                              const std::string& prefix) {
    float pos_fill = fill_value_from_name(prefix + "embed_positions.weight");
    auto pos_embed_table =
        ov::opset11::Constant::create(precision,
                                      ov::Shape{max_target_positions, hidden_size},
                                      std::vector<float>(max_target_positions * hidden_size, pos_fill));
    pos_embed_table->set_friendly_name(prefix + "embed_positions.weight");

    auto pos_ids_i32 = std::make_shared<ov::op::v0::Convert>(position_ids, ov::element::i32);
    pos_ids_i32->set_friendly_name(prefix + "pos_ids_convert");
    auto pos_embed =
        std::make_shared<ov::opset11::Gather>(pos_embed_table,
                                              pos_ids_i32,
                                              ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0}),
                                              0);
    pos_embed->set_friendly_name(prefix + "pos_embed_gather");

    auto hidden_states = std::make_shared<ov::opset11::Add>(token_embed, pos_embed);
    hidden_states->set_friendly_name(prefix + "embed_add");

    return hidden_states->output(0);
}

std::shared_ptr<ov::Model> ModelBuilder::build_whisper_decoder(const ModelConfig& config) {
    clear();
    const auto prec = config.precision;
    const auto d = config.hidden_size;
    const auto heads = config.num_heads;
    const auto hd = config.head_dim;

    auto input_ids = parameter(ov::element::i64, ov::PartialShape{-1, -1}, "input_ids");
    auto encoder_hidden_states =
        parameter(ov::element::f32, ov::PartialShape{-1, -1, static_cast<int64_t>(d)}, "encoder_hidden_states");
    auto beam_idx = parameter(ov::element::i32, ov::PartialShape{-1}, "beam_idx");

    ov::Output<ov::Node> enc_hs = encoder_hidden_states->output(0);
    if (prec != ov::element::f32) {
        auto cvt = std::make_shared<ov::op::v0::Convert>(enc_hs, prec);
        cvt->set_friendly_name("model.decoder.enc_hs_convert");
        enc_hs = cvt->output(0);
    }

    auto token_embed = make_embedding(input_ids->output(0), config.vocab_size, d, "model.decoder.embed_tokens", prec);

    // Pre-build layer 0's key read state — NPUW matchers need kv_seq_len from ShapeOf(beam_gather)[2]
    auto layer0_k_read = make_kv_cache_read(input_ids->output(0),
                                            beam_idx->output(0),
                                            heads,
                                            hd,
                                            make_kv_var_id("0", ".decoder.", "key"),
                                            prec);

    auto cache_pos = make_cache_position_ids(input_ids->output(0), layer0_k_read.beam_gather, "model.decoder.");
    auto hidden_states = make_whisper_positional_embedding(token_embed,
                                                           cache_pos.position_ids,
                                                           config.max_target_positions,
                                                           d,
                                                           prec,
                                                           "model.decoder.");
    auto shared_mask = make_whisper_causal_mask(cache_pos, "model.decoder.");

    // Self-attention (layer-0 reuses pre-built key Variable)
    Attention self_attn{};
    self_attn.hidden_size = d;
    self_attn.num_heads = heads;
    self_attn.num_kv_heads = heads;
    self_attn.head_dim = hd;
    self_attn.precision = prec;
    self_attn.weight_fn = config.weight;
    self_attn.bias_fn = config.attn_bias;
    self_attn.o_proj_name = "self_attn.out_proj";
    self_attn.sdpa_mask = shared_mask;

    self_attn.kv_cache_fn = [&](const ov::Output<ov::Node>& k,
                                const ov::Output<ov::Node>& v,
                                size_t layer) -> std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> {
        auto layer_str = std::to_string(layer);
        KVCacheResult k_cache;
        if (layer == 0) {
            auto k_var_id = make_kv_var_id(layer_str, ".decoder.", "key");
            auto k_concat = std::make_shared<ov::opset11::Concat>(ov::OutputVector{layer0_k_read.beam_gather, k}, 2);
            k_concat->set_friendly_name(k_var_id + "_concat");
            auto k_assign = std::make_shared<ov::op::v6::Assign>(k_concat, layer0_k_read.variable);
            k_assign->set_friendly_name(k_var_id + "_assign");
            k_cache = {k_concat->output(0), layer0_k_read.beam_gather, k_assign};
        } else {
            k_cache = make_kv_cache_concat(k,
                                           input_ids->output(0),
                                           beam_idx->output(0),
                                           heads,
                                           hd,
                                           make_kv_var_id(layer_str, ".decoder.", "key"),
                                           prec);
        }
        auto v_cache = make_kv_cache_concat(v,
                                            input_ids->output(0),
                                            beam_idx->output(0),
                                            heads,
                                            hd,
                                            make_kv_var_id(layer_str, ".decoder.", "value"),
                                            prec);
        m_sinks.push_back(std::dynamic_pointer_cast<ov::op::Sink>(k_cache.assign));
        m_sinks.push_back(std::dynamic_pointer_cast<ov::op::Sink>(v_cache.assign));
        return {k_cache.concatenated, v_cache.concatenated};
    };

    // Cross-attention (store-only encoder KV cache)
    Attention cross_attn{};
    cross_attn.hidden_size = d;
    cross_attn.num_heads = heads;
    cross_attn.num_kv_heads = heads;
    cross_attn.head_dim = hd;
    cross_attn.precision = prec;
    cross_attn.weight_fn = config.weight;
    cross_attn.bias_fn = config.attn_bias;
    cross_attn.o_proj_name = "encoder_attn.out_proj";
    cross_attn.attn_prefix = "encoder_attn.";

    cross_attn.kv_cache_fn = [&](const ov::Output<ov::Node>& k,
                                 const ov::Output<ov::Node>& v,
                                 size_t layer) -> std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> {
        auto layer_str = std::to_string(layer);
        auto k_cache = make_encoder_kv_cache(k, heads, hd, make_kv_var_id(layer_str, ".encoder.", "key"), prec);
        auto v_cache = make_encoder_kv_cache(v, heads, hd, make_kv_var_id(layer_str, ".encoder.", "value"), prec);
        m_sinks.push_back(std::dynamic_pointer_cast<ov::op::Sink>(k_cache.assign));
        m_sinks.push_back(std::dynamic_pointer_cast<ov::op::Sink>(v_cache.assign));
        return {k_cache.concatenated, v_cache.concatenated};
    };

    auto current = make_transformer_layers(
        hidden_states,
        config.get_decoder_layers(),
        "model.decoder.layers.",
        [&](const ov::Output<ov::Node>& input, const std::string& prefix, size_t layer) {
            auto call_self = [&](const ov::Output<ov::Node>& normed, const std::string& pfx) {
                return self_attn(normed, {}, pfx, layer);
            };
            auto call_cross = [&](const ov::Output<ov::Node>& normed, const std::string& pfx) {
                return cross_attn(normed, enc_hs, pfx, layer);
            };

            return make_cross_attn_decoder_layer(input, config.norm, call_self, call_cross, config.ffn, prefix);
        });

    auto final_norm = config.norm(current, "model.decoder.layer_norm");
    auto logits = make_lm_head(final_norm, d, config.vocab_size, "proj_out", prec, config.weight);

    // Always f32 output — WhisperPipeline reads logits as f32
    ov::Output<ov::Node> logits_out = logits;
    if (prec != ov::element::f32) {
        auto cvt = std::make_shared<ov::op::v0::Convert>(logits, ov::element::f32);
        cvt->set_friendly_name("model.decoder.logits_convert");
        logits_out = cvt->output(0);
    }

    return make_model(logits_out, "logits", "synthetic_whisper_decoder");
}

std::shared_ptr<ov::Model> ModelBuilder::build_embedding_encoder(const ModelConfig& config) {
    clear();

    const auto prec = config.precision;
    const auto hs = config.hidden_size;

    auto input_ids = parameter(ov::element::i64, ov::PartialShape{-1, -1}, "input_ids");
    auto attention_mask = parameter(ov::element::i64, ov::PartialShape{-1, -1}, "attention_mask");
    auto token_type_ids = parameter(ov::element::i64, ov::PartialShape{-1, -1}, "token_type_ids");

    auto word_embed = make_embedding(input_ids->output(0), config.vocab_size, hs, "embeddings.word_embeddings", prec);

    auto ids_shape = std::make_shared<ov::opset11::ShapeOf>(input_ids, ov::element::i64);
    auto idx1 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto axis0 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto seq_len = std::make_shared<ov::opset11::Gather>(ids_shape, idx1, axis0);
    auto start = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto step = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto pos_ids = std::make_shared<ov::op::v4::Range>(start, seq_len, step, ov::element::i64);
    pos_ids->set_friendly_name("embeddings.position_ids");
    auto pos_embed =
        make_embedding(pos_ids->output(0), config.max_position_embeddings, hs, "embeddings.position_embeddings", prec);

    auto type_embed =
        make_embedding(token_type_ids->output(0), config.type_vocab_size, hs, "embeddings.token_type_embeddings", prec);

    auto embed_sum1 = std::make_shared<ov::opset11::Add>(word_embed, pos_embed);
    embed_sum1->set_friendly_name("embeddings.add_pos");
    auto embed_sum2 = std::make_shared<ov::opset11::Add>(embed_sum1, type_embed);
    embed_sum2->set_friendly_name("embeddings.add_type");
    auto embed_normed = config.norm(embed_sum2->output(0), "embeddings.LayerNorm");

    auto sdpa_mask = make_padding_mask(attention_mask->output(0), prec);

    Attention bert_attn{};
    bert_attn.hidden_size = hs;
    bert_attn.num_heads = config.num_heads;
    bert_attn.num_kv_heads = config.num_heads;
    bert_attn.head_dim = config.head_dim;
    bert_attn.precision = prec;
    bert_attn.weight_fn = config.weight;
    bert_attn.bias_fn = config.attn_bias;
    bert_attn.sdpa_mask = sdpa_mask;

    auto current =
        make_transformer_layers(embed_normed,
                                config.num_layers,
                                "encoder.layer.",
                                [&](const ov::Output<ov::Node>& input, const std::string& prefix, size_t /*layer*/) {
                                    return make_post_norm_layer(
                                        input,
                                        config.norm,
                                        [&](const ov::Output<ov::Node>& inp, const std::string& pfx) {
                                            return bert_attn(inp, {}, pfx);
                                        },
                                        config.ffn,
                                        prefix);
                                });

    return make_model(current, "last_hidden_state", "synthetic_encoder_model");
}

}  // namespace npuw
}  // namespace test
}  // namespace ov
