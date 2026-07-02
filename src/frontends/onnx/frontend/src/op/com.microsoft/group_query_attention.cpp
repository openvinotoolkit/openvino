// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/group_query_attention.hpp"

#include <algorithm>

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/transpose.hpp"
#include "utils/attention.hpp"
#include "utils/common.hpp"
#include "utils/split.hpp"

using namespace ov::op;

namespace ov::frontend::onnx::com_microsoft {

namespace detail {
using ov::frontend::onnx::attention::get_dimensions;

// Reshape a per-channel KV scale [kv_num_heads * head_dim] to [1, kv_num_heads, 1, head_dim] so it
// broadcasts over the [B, kv_num_heads, S, head_dim] KV layout used inside GroupQueryAttention.
//
// Built as a fresh Constant that owns a copy of this layer's scale data (the memcpy ctor), giving a
// unique data pointer and no inherited weightless-cache attribute. This is required for NPUW FOLD:
// a shape-Constant or buffer-sharing clone would be deduped/aliased across the identical layers and
// break per-layer weight-bank matching during compile.
inline ov::Output<ov::Node> reshape_scale_4d(const ov::Output<ov::Node>& scale,
                                             int64_t kv_num_heads,
                                             const std::string& base) {
    auto scale_const = ov::as_type_ptr<v0::Constant>(scale.get_node_shared_ptr());
    const auto& ps = scale.get_partial_shape();
    if (scale_const && ps.is_static()) {
        const int64_t total = static_cast<int64_t>(ov::shape_size(ps.to_shape()));
        const int64_t head_dim = total / kv_num_heads;
        const ov::Shape new_shape{1, static_cast<size_t>(kv_num_heads), 1, static_cast<size_t>(head_dim)};
        auto reshaped = std::make_shared<v0::Constant>(scale_const->get_element_type(),
                                                       new_shape,
                                                       scale_const->get_data_ptr());
        reshaped->set_friendly_name(base + "/scale_4d");
        return reshaped;
    }
    // Fallback for a non-constant scale (not expected for baked-initializer scales).
    const auto target = v0::Constant::create(ov::element::i64,
                                             ov::Shape{4},
                                             std::vector<int64_t>{int64_t{1}, kv_num_heads, int64_t{1}, int64_t{-1}});
    target->set_friendly_name(base + "/scale_shape");
    auto r = std::make_shared<v1::Reshape>(scale, target, false);
    r->set_friendly_name(base + "/scale_reshape");
    return r;
}

// Symmetric per-channel dequantize of an int8 KV tensor: Convert(int8->f32) * scale.
// Each call clones its own single-reader scale Constant (see reshape_scale_4d).
inline ov::Output<ov::Node> dequantize_kv(const ov::Output<ov::Node>& quant,
                                          const ov::Output<ov::Node>& scale,
                                          int64_t kv_num_heads,
                                          const ov::element::Type& compute_type,
                                          const std::string& base) {
    auto q_f = std::make_shared<v0::Convert>(quant, ov::element::f32);
    q_f->set_friendly_name(base + "/dq_convert_f32");
    auto scale4d = reshape_scale_4d(scale, kv_num_heads, base + "/dq");
    auto mul = std::make_shared<v1::Multiply>(q_f, scale4d);
    mul->set_friendly_name(base + "/dq_mul");
    ov::Output<ov::Node> deq = mul;
    if (compute_type != ov::element::f32) {
        auto cvt = std::make_shared<v0::Convert>(deq, compute_type);
        cvt->set_friendly_name(base + "/dq_convert_compute");
        deq = cvt;
    }
    return deq;
}

// Symmetric per-channel requantize of the present KV back to int8: round(value / scale) clamped to
// [-128, 127], so the present_* graph outputs keep their declared int8 type. Own single-reader scale.
inline ov::Output<ov::Node> quantize_kv(const ov::Output<ov::Node>& value,
                                        const ov::Output<ov::Node>& scale,
                                        int64_t kv_num_heads,
                                        const std::string& base) {
    ov::Output<ov::Node> v_f = value;
    if (value.get_element_type() != ov::element::f32) {
        auto cvt = std::make_shared<v0::Convert>(value, ov::element::f32);
        cvt->set_friendly_name(base + "/q_convert_f32");
        v_f = cvt;
    }
    auto scale4d = reshape_scale_4d(scale, kv_num_heads, base + "/q");
    auto scaled = std::make_shared<v1::Divide>(v_f, scale4d);
    scaled->set_friendly_name(base + "/q_div");
    auto rounded = std::make_shared<v5::Round>(scaled, v5::Round::RoundMode::HALF_TO_EVEN);
    rounded->set_friendly_name(base + "/q_round");
    auto clamped = std::make_shared<v0::Clamp>(rounded, -128.0, 127.0);
    clamped->set_friendly_name(base + "/q_clamp");
    auto out = std::make_shared<v0::Convert>(clamped, ov::element::i8);
    out->set_friendly_name(base + "/q_convert_i8");
    return out;
}
}  // namespace detail

namespace opset_1 {
ov::OutputVector group_query_attention(const ov::frontend::onnx::Node& node) {
    constexpr size_t inputs_count_min = 7;   // Taken from ONNX spec
    constexpr size_t inputs_count_max = 14;  // Taken from ONNX spec

    // Minimum required inputs basing on the spec and ONNX Runtime code: 7
    // 0: packed QKV (mandatory)
    // 3-4: possibly null (if unused)
    // 5: seqlens_k (mandatory)
    // 6: total_sequence_length (mandatory in the spec)
    common::default_op_checks(node, inputs_count_min, inputs_count_max);

    const auto onnx_op_inputs = node.get_ov_inputs();
    const auto num_heads = node.get_attribute_value<int64_t>("num_heads");
    const auto kv_num_heads = node.get_attribute_value<int64_t>("kv_num_heads");
    const auto scale = node.get_attribute_value<float>("scale", 0.0f);
    const auto do_rotary = node.get_attribute_value<int64_t>("do_rotary", 0);
    const auto rotary_interleaved = node.get_attribute_value<int64_t>("rotary_interleaved", 0);

    // int8 KV-cache mode: KV is stored int8 and the op carries kv_cache_bit_width plus per-channel
    // scale inputs (key=12, value=13). The internal op only supports f16/f32 KV, so dequantize past
    // KV before it and requantize present KV after it, leaving the int8 KV graph I/O unchanged.
    const auto kv_cache_bit_width = node.get_attribute_value<int64_t>("kv_cache_bit_width", 0);
    constexpr size_t k_scale_index = 12;
    constexpr size_t v_scale_index = 13;
    constexpr size_t kv_op_inputs_end = 9;  // op consumes inputs 0..8 (Q,K,V,pastK,pastV,seqlens,total,cos,sin)
    const bool int8_kv_cache = kv_cache_bit_width > 0 &&
                               common::is_input_valid(onnx_op_inputs, k_scale_index) &&
                               common::is_input_valid(onnx_op_inputs, v_scale_index);
    // Layer-scoped base name for the inserted dequant/requant nodes (per-layer unique for NPUW FOLD).
    const std::string gqa_name = !node.get_name().empty() ? node.get_name() : node.output(0);
    if (int8_kv_cache) {
        FRONT_END_OP_CONVERSION_CHECK(kv_cache_bit_width == 8,
                                      "GroupQueryAttention: only 8-bit int KV cache is supported.");
        const auto k_qt = node.get_attribute_value<std::string>("k_quant_type", "PER_CHANNEL");
        const auto v_qt = node.get_attribute_value<std::string>("v_quant_type", "PER_CHANNEL");
        FRONT_END_OP_CONVERSION_CHECK(k_qt == "PER_CHANNEL" && v_qt == "PER_CHANNEL",
                                      "GroupQueryAttention: only PER_CHANNEL int8 KV quantization is supported, got "
                                      "k_quant_type=",
                                      k_qt,
                                      ", v_quant_type=",
                                      v_qt);
    }

    if (0 != do_rotary) {
        constexpr size_t cos_cache_index = 7;
        constexpr size_t sin_cache_index = 8;

        FRONT_END_OP_CONVERSION_CHECK(common::is_input_valid(onnx_op_inputs, sin_cache_index) &&
                                          common::is_input_valid(onnx_op_inputs, cos_cache_index),
                                      "GroupQueryAttention: cos_cache and sin_cache inputs are required when "
                                      "do_rotary is enabled.");
    }

    // In ONNX, the format of input QKV is [B, S, N*H] and of past_kv is [B, N, S, H]
    // In OV, we always use [B, N, S, H]
    auto perm = v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});

    auto Q = onnx_op_inputs[0];
    auto K = onnx_op_inputs[1];
    auto V = onnx_op_inputs[2];

    FRONT_END_OP_CONVERSION_CHECK(!ov::op::util::is_null(Q), "GroupQueryAttention: Expecting Q/QKV not null.");

    const auto& seqlens_k = onnx_op_inputs[5];

    FRONT_END_OP_CONVERSION_CHECK(!ov::op::util::is_null(seqlens_k),
                                  "GroupQueryAttention: Expecting seqlens_k not null.");

    const auto q_shape_node = std::make_shared<v3::ShapeOf>(Q);
    const auto batch_size_node = detail::get_dimensions(q_shape_node, {0});
    const auto current_seqlen_size_node = detail::get_dimensions(q_shape_node, {1});
    const auto hidden_size_node = detail::get_dimensions(q_shape_node, {2});

    OutputVector ov_op_inputs;
    if (ov::op::util::is_null(K) && ov::op::util::is_null(V)) {
        auto total_num_heads_node =
            v0::Constant::create(ov::element::i64, ov::Shape{1}, {num_heads + kv_num_heads + kv_num_heads});
        auto head_size_node = std::make_shared<v1::Divide>(hidden_size_node, total_num_heads_node);
        auto packed_qkv_shape = std::make_shared<v0::Concat>(
            ov::NodeVector{batch_size_node, current_seqlen_size_node, total_num_heads_node, head_size_node},
            0);

        auto inputs_qkv = std::make_shared<v1::Reshape>(Q, packed_qkv_shape, false)->output(0);
        inputs_qkv = std::make_shared<v1::Transpose>(inputs_qkv, perm);
        auto split = ov::op::util::make_split(inputs_qkv, {num_heads, kv_num_heads, kv_num_heads}, 1);

        std::copy(split.begin(), split.end(), std::back_inserter(ov_op_inputs));

        FRONT_END_OP_CONVERSION_CHECK(ov_op_inputs.size() == 3,
                                      "GroupQueryAttention: Expecting QKV split to produce 3 outputs.");
    } else {
        FRONT_END_OP_CONVERSION_CHECK(!ov::op::util::is_null(K), "GroupQueryAttention: Expecting K not null.");
        FRONT_END_OP_CONVERSION_CHECK(!ov::op::util::is_null(V), "GroupQueryAttention: Expecting V not null.");

        auto num_heads_node = v0::Constant::create(ov::element::i64, ov::Shape{1}, {num_heads});
        auto head_size_node = std::make_shared<v1::Divide>(hidden_size_node, num_heads_node);
        auto q_shape = std::make_shared<v0::Concat>(
            ov::NodeVector{batch_size_node, current_seqlen_size_node, num_heads_node, head_size_node},
            0);

        Q = std::make_shared<v1::Reshape>(Q, q_shape, false)->output(0);
        Q = std::make_shared<v1::Transpose>(Q, perm);
        ov_op_inputs.push_back(std::move(Q));

        auto kv_num_heads_node = v0::Constant::create(ov::element::i64, ov::Shape{1}, {kv_num_heads});
        auto kv_shape = std::make_shared<v0::Concat>(
            ov::NodeVector{batch_size_node, current_seqlen_size_node, kv_num_heads_node, head_size_node},
            0);

        K = std::make_shared<v1::Reshape>(K, kv_shape, false)->output(0);
        V = std::make_shared<v1::Reshape>(V, kv_shape, false)->output(0);
        K = std::make_shared<v1::Transpose>(K, perm);
        V = std::make_shared<v1::Transpose>(V, perm);
        ov_op_inputs.push_back(std::move(K));
        ov_op_inputs.push_back(std::move(V));
    }

    // Forward the remaining op inputs (past_key=3, past_value=4, seqlens=5, total=6, cos=7, sin=8).
    // For an int8 KV cache, dequantize past_key/past_value to the compute type so the internal op
    // (and the downstream NPUW decomposition) sees plain f16/f32 KV, exactly like the fp16 model.
    // The trailing quantization inputs (scales at 12/13) are consumed here, not forwarded to the op.
    const auto compute_type = ov_op_inputs[0].get_element_type();

    // Forward op inputs 3..8; in int8 mode dequantize past_key/past_value (3/4) and drop the
    // trailing scale inputs (the op itself only takes inputs 0..8).
    const size_t forward_end = int8_kv_cache ? std::min(onnx_op_inputs.size(), kv_op_inputs_end) : onnx_op_inputs.size();
    for (size_t i = ov_op_inputs.size(); i < forward_end; ++i) {
        if (int8_kv_cache && (i == 3 || i == 4)) {
            const auto& kv_scale = (i == 3) ? onnx_op_inputs[k_scale_index] : onnx_op_inputs[v_scale_index];
            const std::string base = gqa_name + (i == 3 ? "/past_key" : "/past_value");
            ov_op_inputs.push_back(detail::dequantize_kv(onnx_op_inputs[i], kv_scale, kv_num_heads, compute_type, base));
        } else {
            ov_op_inputs.push_back(onnx_op_inputs[i]);
        }
    }

    if (int8_kv_cache) {
        // Requantize the present_key/present_value (op outputs 1/2) back to int8.
        auto gqa = std::make_shared<internal::GroupQueryAttention>(ov_op_inputs,
                                                                   num_heads,
                                                                   kv_num_heads,
                                                                   scale,
                                                                   do_rotary,
                                                                   rotary_interleaved);
        const auto gqa_outputs = gqa->outputs();
        return {gqa_outputs[0],
                detail::quantize_kv(gqa_outputs[1], onnx_op_inputs[k_scale_index], kv_num_heads,
                                    gqa_name + "/present_key"),
                detail::quantize_kv(gqa_outputs[2], onnx_op_inputs[v_scale_index], kv_num_heads,
                                    gqa_name + "/present_value")};
    }

    for (size_t i = ov_op_inputs.size(); i < onnx_op_inputs.size(); ++i) {
        ov_op_inputs.push_back(onnx_op_inputs[i]);
    }

    return std::make_shared<internal::GroupQueryAttention>(ov_op_inputs,
                                                           num_heads,
                                                           kv_num_heads,
                                                           scale,
                                                           do_rotary,
                                                           rotary_interleaved)
        ->outputs();
}

ONNX_OP("GroupQueryAttention", OPSET_SINCE(1), com_microsoft::opset_1::group_query_attention, MICROSOFT_DOMAIN);

}  // namespace opset_1

}  // namespace ov::frontend::onnx::com_microsoft
