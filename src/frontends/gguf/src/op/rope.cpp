// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/decompositions/rope.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/node_registry.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

OutputVector translate_rope(const NodeContext& context) {
    num_inputs_check(context, 2, 3);

    ov::Output<Node> res;

    auto data = context.get_input(0);
    const auto data_shape = data.get_partial_shape();
    FRONT_END_OP_CONVERSION_CHECK(data_shape.rank().is_static() && data_shape.rank().get_length() == 4 &&
                                      data_shape[2].is_static() && data_shape[3].is_static(),
                                  "ROPE requires a rank-4 [batch, tokens, heads, head_size] operand "
                                  "with static head dimensions");
    int64_t head_size = data_shape[3].get_length();
    auto rope_config = context.get_attribute<RopeConfig>("rope_config");
    constexpr int TYPE_NORMAL = 0;
    constexpr int TYPE_NEOX = 1;
    constexpr int TYPE_IMROPE = 2;

    constexpr int TYPE_VISION = 3;
    constexpr int TYPE_MROPE = 4;

    const auto explicit_mode = context.get_attribute<std::string>("rope_mode", "normal");
    FRONT_END_CHECK_IMPLEMENTED(explicit_mode == "normal" || explicit_mode == "neox" || explicit_mode == "imrope" ||
                                    explicit_mode == "mrope" || explicit_mode == "vision",
                                "Unsupported ROPE mode: " + explicit_mode);
    const int mode = explicit_mode == "normal"   ? TYPE_NORMAL
                     : explicit_mode == "neox"   ? TYPE_NEOX
                     : explicit_mode == "imrope" ? TYPE_IMROPE
                     : explicit_mode == "mrope"  ? TYPE_MROPE
                                                 : TYPE_VISION;

    if (mode == TYPE_VISION) {
        // ggml vision RoPE rotates the full head in two halves; n_dims is half the
        // head width, and each position section restarts its frequency progression.
        using namespace ov::op;
        const auto shape = data.get_partial_shape();
        FRONT_END_OP_CONVERSION_CHECK(shape.rank() == 4 && shape[3].is_static(),
                                      "Vision RoPE requires rank four and a static head width");
        const int64_t half = shape[3].get_length() / 2;
        FRONT_END_OP_CONVERSION_CHECK(2 * half == shape[3].get_length() && rope_config.n_dims == half,
                                      "Vision RoPE n_dims must equal half the head width");
        FRONT_END_OP_CONVERSION_CHECK(rope_config.ext_factor == 0.f && context.get_input_size() == 2,
                                      "Vision RoPE with YaRN or frequency-factor weights is not supported");
        auto sections = rope_config.sections;
        int64_t total = 0;  // four int32 counts always fit in 64 bits
        for (auto section : sections) {
            FRONT_END_OP_CONVERSION_CHECK(section >= 0, "Negative vision RoPE section");
            total += section;
        }
        FRONT_END_OP_CONVERSION_CHECK(total > 0 && total <= 2 * half && rope_config.freq_base > 0.f,
                                      "Invalid vision RoPE sections or frequency base");
        std::vector<int64_t> axes(half);
        std::vector<float> factors(half);
        for (int64_t i = 0; i < half; ++i) {
            int offset = int(i % total), section = 0;
            while (section < 3 && offset >= sections[section]) {
                offset -= sections[section];
                ++section;
            }
            axes[i] = section;
            factors[i] = std::pow(rope_config.freq_base, -2.f * offset / half) * rope_config.freq_scale;
        }
        auto positions = std::make_shared<v1::Reshape>(context.get_input(1),
                                                       v0::Constant::create(ov::element::i64, {2}, {4, -1}),
                                                       false);
        auto transposed =
            std::make_shared<v1::Transpose>(positions, v0::Constant::create(ov::element::i64, {2}, {1, 0}));
        auto selected = std::make_shared<v8::Gather>(transposed,
                                                     v0::Constant::create(ov::element::i64, {size_t(half)}, axes),
                                                     v0::Constant::create(ov::element::i64, {}, {1}));
        auto theta = std::make_shared<v1::Multiply>(std::make_shared<v0::Convert>(selected, ov::element::f32),
                                                    v0::Constant::create(ov::element::f32, {size_t(half)}, factors));
        auto expand = v0::Constant::create(ov::element::i64, {2}, {0, 2});
        auto scale = v0::Constant::create(ov::element::f32, {}, {rope_config.attn_factor});
        auto cos =
            std::make_shared<v1::Multiply>(std::make_shared<v0::Unsqueeze>(std::make_shared<v0::Cos>(theta), expand),
                                           scale);
        auto sin =
            std::make_shared<v1::Multiply>(std::make_shared<v0::Unsqueeze>(std::make_shared<v0::Sin>(theta), expand),
                                           scale);
        const auto type = data.get_element_type();
        if (type != ov::element::f32)
            data = std::make_shared<v0::Convert>(data, ov::element::f32);
        ov::pass::NodeRegistry reg;
        res = ov::decomposition::rope(reg, data, cos, sin, half);
        if (type != ov::element::f32)
            res = std::make_shared<v0::Convert>(res, type);
        return rename_outputs_with_suffix({res}, context.get_name());
    }

    Output<Node> cos_theta_node;
    Output<Node> sin_theta_node;
    {
        auto inp_pos = context.get_input(1).get_node_shared_ptr();
        std::shared_ptr<ov::Node> rope_freqs_weight;
        if (context.get_input_size() == 3) {
            rope_freqs_weight = context.get_input(2).get_node_shared_ptr();
        }
        // Layers that share a position tensor, frequency factors and RoPE parameters share one
        // sin/cos table instead of each rebuilding the same subgraph.
        const auto key = [&] {
            std::ostringstream stream;
            stream << "rope:" << static_cast<const void*>(inp_pos.get()) << ':'
                   << static_cast<const void*>(rope_freqs_weight.get()) << ':' << rope_config.n_dims << ':'
                   << rope_config.n_ctx_orig << ':' << rope_config.freq_base << ':' << rope_config.freq_scale << ':'
                   << rope_config.ext_factor << ':' << rope_config.attn_factor << ':' << rope_config.beta_fast << ':'
                   << rope_config.beta_slow << ':' << mode;
            for (const auto section : rope_config.sections) {
                stream << ':' << section;
            }
            return stream.str();
        }();
        if (context.has_cached(key + "/sin") && context.has_cached(key + "/cos")) {
            sin_theta_node = context.get_cached(key + "/sin");
            cos_theta_node = context.get_cached(key + "/cos");
        } else {
            auto sin_cos = make_sin_cos(rope_config,
                                        inp_pos,
                                        rope_freqs_weight,
                                        mode == TYPE_IMROPE,
                                        false,
                                        mode == TYPE_MROPE);
            sin_theta_node = sin_cos.first;
            cos_theta_node = sin_cos.second;
            context.set_cached(key + "/sin", sin_theta_node);
            context.set_cached(key + "/cos", cos_theta_node);
        }
    }

    const auto output_type = data.get_element_type();
    if (data.get_element_type() != ov::element::f32) {
        data = std::make_shared<ov::op::v0::Convert>(data, ov::element::f32);
    }

    // "rope_offset" leaves the first channels of every head unrotated and rotates [offset, head_size).
    const int64_t offset = context.get_attribute<int64_t>("rope_offset", 0);
    {
        const int64_t n_rot = rope_config.n_dims > 0 ? rope_config.n_dims : head_size - offset;
        FRONT_END_OP_CONVERSION_CHECK(offset >= 0 && offset % 2 == 0 && n_rot > 0 && n_rot % 2 == 0 &&
                                          offset <= head_size && n_rot <= head_size - offset,
                                      "ROPE requires even dimensions and offset within the head");
    }
    Output<Node> prefix;
    if (offset != 0) {
        auto axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto start = ov::op::v0::Constant::create(ov::element::i64, {1}, {offset});
        auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {head_size});
        prefix = std::make_shared<ov::op::v8::Slice>(data, zero, start, one, axis);
        data = std::make_shared<ov::op::v8::Slice>(data, start, end, one, axis);
        head_size -= offset;
    }

    if (mode == TYPE_NORMAL) {
        // Emit the Flux-style interleaved RoPE pattern so ov::pass::RoPEFusionFlux
        // folds this subgraph into ov::op::internal::RoPE → GPU ocl::rope::opt kernel.
        // RoPEFusionFlux requires rank-4 x with static last two dims [n_heads, head_size].
        // After the VIEW prologue the data is already [B,L,n_heads,head_size].
        const int64_t n_rot = rope_config.n_dims > 0 ? rope_config.n_dims : head_size;
        const int64_t half = n_rot / 2;

        ov::Output<ov::Node> rotary_data = data;
        ov::Output<ov::Node> pass_through;
        if (n_rot < head_size) {
            auto axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
            auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            auto n_rot_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_rot});
            auto head_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {head_size});
            rotary_data = std::make_shared<ov::op::v8::Slice>(data, zero, n_rot_c, one, axis);
            pass_through = std::make_shared<ov::op::v8::Slice>(data, n_rot_c, head_c, one, axis);
        }

        // Split only the innermost axis. Copying the first three dimensions from the operand
        // preserves both ordinary [B,L,H,S] and token-major [L,B,H,S] runtime layouts.
        auto paired_shape =
            ov::op::v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{0, 0, 0, half, 2});
        auto x_paired = std::make_shared<ov::op::v1::Reshape>(rotary_data, paired_shape, true);

        auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1LL});
        auto data_split = std::make_shared<ov::op::v1::Split>(x_paired, split_axis, 2);
        auto x0 = data_split->output(0);
        auto x1 = data_split->output(1);

        auto neg_one_f = ov::op::v0::Constant::create(data.get_element_type(), ov::Shape{}, {-1.0f});
        auto x1_neg = std::make_shared<ov::op::v1::Multiply>(x1, neg_one_f);

        auto x_rotated_paired = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{x1_neg, x0}, -1);
        auto rotary_shape =
            ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 0, 0, n_rot});
        auto x_rotated = std::make_shared<ov::op::v1::Reshape>(x_rotated_paired, rotary_shape, true);

        // Expand cos/sin from [B, L, 1, half] to [B, L, 1, head_size].
        auto expand_cos_sin = [&](ov::Output<ov::Node> cs) -> ov::Output<ov::Node> {
            auto cs_unsq =
                std::make_shared<ov::op::v0::Unsqueeze>(cs,
                                                        ov::op::v0::Constant::create(ov::element::i64, {1}, {-1LL}));
            auto bcast_target =
                ov::op::v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{1, 1, 1, half, 2});
            auto bcast =
                std::make_shared<ov::op::v3::Broadcast>(cs_unsq, bcast_target, ov::op::BroadcastType::BIDIRECTIONAL);
            auto flat = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 0, 0, n_rot});
            return std::make_shared<ov::op::v1::Reshape>(bcast, flat, /*special_zero=*/true);
        };
        auto cos_full = expand_cos_sin(cos_theta_node);
        auto sin_full = expand_cos_sin(sin_theta_node);

        auto y1 = std::make_shared<ov::op::v1::Multiply>(rotary_data, cos_full);
        auto y2 = std::make_shared<ov::op::v1::Multiply>(x_rotated, sin_full);
        ov::Output<ov::Node> rotated = std::make_shared<ov::op::v1::Add>(y1, y2);
        res = n_rot < head_size ? std::make_shared<ov::op::v0::Concat>(ov::OutputVector{rotated, pass_through}, -1)
                                : rotated;
    } else if (mode == TYPE_NEOX) {
        // Partial rotary (ggml n_dims < head_dim): only the first n_dims of every head are
        // rotated; the remaining tail is passed through unchanged. cos/sin have width n_dims/2,
        // so the rotated block must be exactly n_dims wide.
        const int64_t head_dim = head_size;
        const int64_t n_rot = rope_config.n_dims > 0 ? rope_config.n_dims : head_dim;

        // Split the head into the rotated block [0, n_rot) and the untouched tail [n_rot, head_dim)
        // on the innermost axis. Both branches below rotate `rotary_in` and re-concatenate the tail.
        auto split_rotary =
            [&](ov::Output<ov::Node> x, ov::Output<ov::Node>& rotary_in, ov::Output<ov::Node>& pass_through) {
                rotary_in = x;
                if (n_rot < head_dim) {
                    auto neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
                    auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
                    auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
                    auto n_rot_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_rot});
                    auto head_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {head_dim});
                    rotary_in = std::make_shared<ov::op::v8::Slice>(x, zero, n_rot_c, one, neg_one);
                    pass_through = std::make_shared<ov::op::v8::Slice>(x, n_rot_c, head_c, one, neg_one);
                }
            };

        // Rotate in the ggml-natural layout, data [B, L, H, S] with cos/sin [B, L, 1, n_rot/2], using the explicit
        // split-halves form: y = x * [cos, cos] + [-x2, x1] * [sin, sin].
        //
        // Do not emit the canonical form that ov::pass::RoPEFusion folds into ov::op::internal::RoPE (the
        // ov::decomposition::rope helper wrapped in transposes to [B, H, L, S]). The GPU plugin's fused RoPE kernel
        // computes this rotation wrongly: stateless GPU graphs of NEOX-rope models (Phi-4-mini, Qwen2.5, Qwen3, ...)
        // turn into garbage after a few decode tokens, while the same graph is correct on CPU.
        ov::Output<ov::Node> rotary_in;
        ov::Output<ov::Node> pass_through;
        split_rotary(data, rotary_in, pass_through);

        const int64_t half = n_rot / 2;
        auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3});
        auto split_lengths = ov::op::v0::Constant::create(ov::element::i64, {2}, std::vector<int64_t>{half, half});
        auto halves = std::make_shared<ov::op::v1::VariadicSplit>(rotary_in, split_axis, split_lengths);
        ov::Output<ov::Node> x1 = halves->output(0);
        ov::Output<ov::Node> x2 = halves->output(1);
        auto x2_neg = std::make_shared<ov::op::v1::Multiply>(
            x2, ov::op::v0::Constant::create(rotary_in.get_element_type(), ov::Shape{}, {-1.0f}));
        auto rotated_half = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{x2_neg, x1}, -1);
        auto cos_full = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{cos_theta_node, cos_theta_node}, -1);
        auto sin_full = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{sin_theta_node, sin_theta_node}, -1);
        ov::Output<ov::Node> roped = std::make_shared<ov::op::v1::Add>(
            std::make_shared<ov::op::v1::Multiply>(rotary_in, cos_full),
            std::make_shared<ov::op::v1::Multiply>(rotated_half, sin_full));
        res = n_rot < head_dim ? std::make_shared<ov::op::v0::Concat>(ov::OutputVector{roped, pass_through}, -1)
                               : roped;
    } else if (mode == TYPE_IMROPE || mode == TYPE_MROPE) {
        // Partial rotary (ggml n_dims < head_dim): only the first n_rot dims of every head are
        // rotated, the tail is passed through unchanged -- e.g. qwen3.5 has head_dim 256 but
        // rope.dimension_count 64. cos/sin carry width n_rot/2, so the rotated block must be
        // exactly n_rot wide; using the full head here rotates the pass-through tail and corrupts
        // every full-attention layer.
        const int64_t head_dim = head_size;
        const int64_t n_rot = rope_config.n_dims > 0 ? rope_config.n_dims : head_dim;

        Output<Node> rotary_in = data;
        Output<Node> pass_through;
        if (n_rot < head_dim) {
            auto neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
            auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            auto n_rot_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_rot});
            auto head_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {head_dim});
            rotary_in = std::make_shared<ov::op::v8::Slice>(data, zero, n_rot_c, one, neg_one);
            pass_through = std::make_shared<ov::op::v8::Slice>(data, n_rot_c, head_c, one, neg_one);
        }

        auto cos_sin_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                    ov::Shape{4},
                                                                    std::vector<int64_t>{0, -1, 1, (n_rot >> 1)});
        auto cos_reshaped = std::make_shared<ov::op::v1::Reshape>(cos_theta_node, cos_sin_shape, true);
        auto sin_reshaped = std::make_shared<ov::op::v1::Reshape>(sin_theta_node, cos_sin_shape, true);

        auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3});
        auto split_a = std::make_shared<ov::op::v1::Split>(rotary_in, split_axis, 2);
        auto x0 = split_a->output(0);
        auto x1 = split_a->output(1);
        auto mul_a = std::make_shared<ov::op::v1::Multiply>(x0, cos_reshaped);
        auto mul_b = std::make_shared<ov::op::v1::Multiply>(x1, sin_reshaped);
        auto sub = std::make_shared<ov::op::v1::Subtract>(mul_a, mul_b);

        auto mul_c = std::make_shared<ov::op::v1::Multiply>(x0, sin_reshaped);
        auto mul_d = std::make_shared<ov::op::v1::Multiply>(x1, cos_reshaped);
        auto add = std::make_shared<ov::op::v1::Add>(mul_c, mul_d);

        Output<Node> rotated = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{sub, add}, 3);
        res = (n_rot < head_dim) ? std::make_shared<ov::op::v0::Concat>(ov::OutputVector{rotated, pass_through}, 3)
                                 : rotated;
    }

    // Fail cleanly on an unmapped mode rather than dereferencing a null res downstream.
    FRONT_END_CHECK_IMPLEMENTED(res.get_node_shared_ptr() != nullptr, "Unsupported ROPE mode");
    if (prefix.get_node_shared_ptr()) {
        res = std::make_shared<ov::op::v0::Concat>(OutputVector{prefix, res}, -1);
    }

    if (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
