// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "model_builder_rope.hpp"

#include <cmath>
#include <vector>

#include "model_builder_internal.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/opsets/opset11.hpp"

namespace ov {
namespace test {
namespace npuw {

namespace {

/// Builds frequency cos/sin chain matching NPUW's AddPositionIdsNode pattern.
struct RoPEFrequencies {
    ov::Output<ov::Node> cos, sin;
};

RoPEFrequencies build_rope_frequencies(size_t head_dim,
                                       ov::element::Type precision,
                                       const ov::Output<ov::Node>& position_ids,
                                       const ov::Output<ov::Node>& shape_source = {},
                                       const std::string& prefix = "model.rope") {
    const size_t half_dim = head_dim / 2;

    std::vector<float> inv_freq_data(half_dim);
    for (size_t i = 0; i < half_dim; ++i) {
        inv_freq_data[i] =
            1.0f / std::pow(kRoPEBaseFrequency, static_cast<float>(2 * i) / static_cast<float>(head_dim));
    }
    auto inv_freq = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1, half_dim, 1}, inv_freq_data);
    inv_freq->set_friendly_name(prefix + ".inv_freq");

    // Broadcast inv_freq to [batch, half_dim, 1] using batch dim from shape_source.
    // This ShapeOf -> Gather -> Concat -> Broadcast chain matches NPUW's RopePatternLLama2.
    const auto& batch_source = shape_source.get_node() ? shape_source : position_ids;
    auto shape_of = std::make_shared<ov::opset11::ShapeOf>(batch_source, ov::element::i64);
    shape_of->set_friendly_name(prefix + ".shapeof");
    auto batch_dim = std::make_shared<ov::opset11::Gather>(
        shape_of,
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {0}),
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    batch_dim->set_friendly_name(prefix + ".batch_gather");
    auto broadcast_shape = std::make_shared<ov::opset11::Concat>(
        ov::OutputVector{
            batch_dim->output(0),
            ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(half_dim)})->output(0),
            ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1})->output(0)},
        0);
    broadcast_shape->set_friendly_name(prefix + ".broadcast_shape");
    auto inv_freq_broadcast =
        std::make_shared<ov::op::v3::Broadcast>(inv_freq, broadcast_shape, ov::op::BroadcastType::BIDIRECTIONAL);
    inv_freq_broadcast->set_friendly_name(prefix + ".inv_freq_broadcast");

    // position_ids [batch, seq] -> Unsqueeze -> Convert(f32) -> MatMul(broadcast_inv_freq)
    // -> Transpose -> Concat(self,self) -> Sin/Cos -> Unsqueeze [batch, 1, seq, head_dim]
    auto unsq_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto unsqueezed = std::make_shared<ov::opset11::Unsqueeze>(position_ids, unsq_axis);
    unsqueezed->set_friendly_name(prefix + ".pos_unsqueeze");

    auto converted = std::make_shared<ov::opset11::Convert>(unsqueezed, ov::element::f32);
    converted->set_friendly_name(prefix + ".pos_convert");

    auto matmul = std::make_shared<ov::opset11::MatMul>(inv_freq_broadcast, converted, false, false);
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

}  // namespace

HalfRotationRoPE::HalfRotationRoPE(size_t hd,
                                   ov::element::Type precision,
                                   const ov::Output<ov::Node>& position_ids,
                                   const ov::Output<ov::Node>& shape_source)
    : head_dim(hd) {
    auto freq = build_rope_frequencies(hd, precision, position_ids, shape_source);
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

InterleavedRoPE::InterleavedRoPE(size_t hd,
                                 ov::element::Type precision,
                                 const ov::Output<ov::Node>& position_ids,
                                 const ov::Output<ov::Node>& shape_source)
    : head_dim(hd) {
    auto freq = build_rope_frequencies(hd, precision, position_ids, shape_source);
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

ov::Output<ov::Node> make_learned_positional_embedding(const ov::Output<ov::Node>& token_embed,
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

}  // namespace npuw
}  // namespace test
}  // namespace ov
