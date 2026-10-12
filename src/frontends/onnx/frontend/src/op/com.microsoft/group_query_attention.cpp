// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/group_query_attention.hpp"

#include <algorithm>

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/transpose.hpp"
#include "utils/attention.hpp"
#include "utils/common.hpp"
#include "utils/split.hpp"

using namespace ov::op;

namespace ov::frontend::onnx::com_microsoft {

namespace detail {
using ov::frontend::onnx::attention::get_dimensions;
}  // namespace detail

namespace opset_1 {
ov::OutputVector group_query_attention(const ov::frontend::onnx::Node& node) {
    constexpr size_t inputs_count_min = 7;   // Taken from ONNX spec
    constexpr size_t inputs_count_max = 16;  // Taken from ONNX spec

    // Minimum required inputs basing on the spec and ONNX Runtime code: 7
    // 0: packed QKV (mandatory)
    // 3-4: possibly null (if unused)
    // 5: seqlens_k (mandatory)
    // 6: total_sequence_length (mandatory in the spec)
    // 12-13: k_scale/v_scale (required when the KV cache is quantized)
    common::default_op_checks(node, inputs_count_min, inputs_count_max);

    const auto onnx_op_inputs = node.get_ov_inputs();
    const auto num_heads = node.get_attribute_value<int64_t>("num_heads");
    const auto kv_num_heads = node.get_attribute_value<int64_t>("kv_num_heads");
    const auto scale = node.get_attribute_value<float>("scale", 0.0f);
    const auto do_rotary = node.get_attribute_value<int64_t>("do_rotary", 0);
    const auto rotary_interleaved = node.get_attribute_value<int64_t>("rotary_interleaved", 0);
    const auto causal = node.get_attribute_value<int64_t>("causal", 1);
    // Quantized KV cache attributes (com.microsoft spec). Default to the unquantized (float KV) behavior.
    const auto kv_cache_bit_width = node.get_attribute_value<int64_t>("kv_cache_bit_width", 0);
    const auto parse_quant_type = [&](const std::string& quant_type_name) {
        using QuantType = ov::op::internal::GroupQueryAttentionQuantType;
        if (quant_type_name == "NONE") {
            return QuantType::NONE;
        }
        if (quant_type_name == "PER_TENSOR") {
            return QuantType::PER_TENSOR;
        }
        if (quant_type_name == "PER_CHANNEL") {
            return QuantType::PER_CHANNEL;
        }
        FRONT_END_GENERAL_CHECK(false, "GroupQueryAttention: unsupported quant type '", quant_type_name, "'.");
        return QuantType::NONE;
    };
    const auto k_quant_type = parse_quant_type(node.get_attribute_value<std::string>("k_quant_type", "NONE"));
    const auto v_quant_type = parse_quant_type(node.get_attribute_value<std::string>("v_quant_type", "NONE"));
    // Sliding-window / softcap / smooth-softmax attributes (com.microsoft spec). Default to no-op values,
    // matching the ONNX Runtime defaults (local_window_size = -1 disables the window).
    const auto local_window_size = node.get_attribute_value<int64_t>("local_window_size", -1);
    const auto sliding_window_cache = node.get_attribute_value<int64_t>("sliding_window_cache", 0);
    const auto softcap = node.get_attribute_value<float>("softcap", 0.0f);
    const auto smooth_softmax = node.get_attribute_value<int64_t>("smooth_softmax", 0);
    const auto qk_output = node.get_attribute_value<int64_t>("qk_output", 0);

    // Reject spec inputs whose semantics are not implemented by the OpenVINO decomposition.
    FRONT_END_OP_CONVERSION_CHECK(
        !common::is_input_valid(onnx_op_inputs, 14) && !common::is_input_valid(onnx_op_inputs, 15),
        "GroupQueryAttention: q_norm_weight/k_norm_weight (QK-Norm) inputs are not "
        "supported.");

    // Reject spec attributes whose semantics are not implemented by the OpenVINO decomposition.
    FRONT_END_OP_CONVERSION_CHECK(causal == 0 || causal == 1,
                                  "GroupQueryAttention: causal must be 0 or 1, got ",
                                  causal,
                                  ".");
    // causal == 0 selects bidirectional attention (no query-relative masking, only the buffer tail beyond
    // total_sequence_length is masked); ONNX Runtime does not allow combining that with a sliding window.
    FRONT_END_OP_CONVERSION_CHECK(causal == 1 || local_window_size == -1,
                                  "GroupQueryAttention: local_window_size requires causal=1, got causal=0 and "
                                  "local_window_size=",
                                  local_window_size,
                                  ".");
    // local_window_size == -1 disables the window; a value >= 1 selects a sliding window. A window of
    // size 0 is an empty attention (every query masks all keys) and is not a valid ONNX Runtime config.
    FRONT_END_OP_CONVERSION_CHECK(local_window_size == -1 || local_window_size >= 1,
                                  "GroupQueryAttention: local_window_size must be -1 (disabled) or >= 1, got ",
                                  local_window_size,
                                  ".");
    // A windowed KV cache requires a real sliding window (local_window_size > 0), matching the ONNX Runtime
    // precondition. batch > 1 is not handled by this decomposition (see the batch_size == 1 check below).
    // Multi-token steps (ORT's "staging" regime) are supported: the decomposition selects the staging vs.
    // in-place cache-write branch from the runtime past/total length, mirroring ORT's PlanWindowedKvCache,
    // so a static multi-token shape is not rejected here.
    if (sliding_window_cache != 0) {
        FRONT_END_OP_CONVERSION_CHECK(local_window_size >= 1,
                                      "GroupQueryAttention: sliding_window_cache=1 requires local_window_size >= 1.");
        FRONT_END_OP_CONVERSION_CHECK(
            common::is_input_valid(onnx_op_inputs, 3) && common::is_input_valid(onnx_op_inputs, 4),
            "GroupQueryAttention: sliding_window_cache=1 requires past_key and past_value.");
        // The windowed cache-end arithmetic uses gap = capacity - local_window_size + 1, which must be >= 1;
        // with capacity < local_window_size it would divide by zero (or a negative gap) at inference. The ONNX
        // Runtime precondition is the same: a cache capacity of at least local_window_size. Enforce it here
        // whenever the past_key capacity (dim 2) is static; a dynamic capacity is clamped in the pass.
        const auto& past_key_ps = onnx_op_inputs[3].get_partial_shape();
        if (past_key_ps.rank().is_static() && past_key_ps.rank().get_length() == 4 && past_key_ps[2].is_static()) {
            FRONT_END_OP_CONVERSION_CHECK(past_key_ps[2].get_length() >= local_window_size,
                                          "GroupQueryAttention: sliding_window_cache=1 requires a past_key/past_value "
                                          "cache capacity (dim 2) of at least local_window_size (",
                                          local_window_size,
                                          "), got capacity = ",
                                          past_key_ps[2].get_length(),
                                          ".");
        }
    }
    FRONT_END_OP_CONVERSION_CHECK(softcap == 0.0f, "GroupQueryAttention: softcap is not supported.");
    // qk_output (spec: emit the QxK' matrix before/after softmax as a 4th `output_qk` output) is not
    // produced by this decomposition. Reject a non-default value rather than silently dropping the output.
    FRONT_END_OP_CONVERSION_CHECK(qk_output == 0, "GroupQueryAttention: qk_output is not supported.");

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

    const auto make_empty_optional_input = []() {
        return v0::Constant::create(ov::element::dynamic, ov::Shape{0}, {})->output(0);
    };

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

        // "Shared KV" (kv_sequence_length == 0): ORT treats this as the past buffer already holding the
        // complete KV, with nothing new appended and K/V skipping RoPE (helper.h). The reshape below sizes
        // K/V using Q's sequence dim (current_seqlen_size_node), so a genuinely empty K/V cannot even be
        // reshaped to it; reject cleanly here instead of failing inside the Reshape with an unrelated
        // element-count-mismatch error.
        const auto& k_ps = K.get_partial_shape();
        FRONT_END_OP_CONVERSION_CHECK(
            !(k_ps.rank().is_static() && k_ps.rank().get_length() == 3 && k_ps[1].is_static() &&
              k_ps[1].get_length() == 0),
            "GroupQueryAttention: kv_sequence_length == 0 (shared KV / past buffer already complete) is not "
            "supported.");

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

    FRONT_END_OP_CONVERSION_CHECK(
        common::is_input_valid(onnx_op_inputs, 3) && common::is_input_valid(onnx_op_inputs, 4),
        "GroupQueryAttention: past_key (input 3) and past_value (input 4) must be provided as tensors");
    // Process optional inputs: use a zero-sized Constant placeholder for missing optional ONNX inputs.
    // Note: When the ONNX's input index changed, the corresponding index in the GroupQueryAttentionInputs enum must
    // also be updated and  may need mapping the index manually.
    for (size_t i = ov_op_inputs.size(); i < inputs_count_max; ++i) {
        if (i < onnx_op_inputs.size() && !ov::op::util::is_null(onnx_op_inputs[i])) {
            ov_op_inputs.push_back(onnx_op_inputs[i]);
        } else {
            ov_op_inputs.push_back(make_empty_optional_input());
        }
    }

    // smooth_softmax's ONNX schema default is -1, not 0, yet ORT's own CPU/CUDA kernels enable it only when
    // the value is exactly 1. Comparing == 1 (not != 0) keeps a graph that never set the attribute from
    // silently getting the smooth-softmax sink path.
    return std::make_shared<internal::GroupQueryAttention>(ov_op_inputs,
                                                           num_heads,
                                                           kv_num_heads,
                                                           scale,
                                                           do_rotary,
                                                           rotary_interleaved,
                                                           kv_cache_bit_width,
                                                           k_quant_type,
                                                           v_quant_type,
                                                           local_window_size,
                                                           sliding_window_cache != 0,
                                                           smooth_softmax == 1,
                                                           causal != 0)
        ->outputs();
}

ONNX_OP("GroupQueryAttention", OPSET_SINCE(1), com_microsoft::opset_1::group_query_attention, MICROSOFT_DOMAIN);

}  // namespace opset_1

}  // namespace ov::frontend::onnx::com_microsoft
