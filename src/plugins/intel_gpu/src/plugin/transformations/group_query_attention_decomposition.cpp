// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "group_query_attention_decomposition.hpp"

#include <limits>
#include <optional>
#include <utility>

#include "intel_gpu/op/sdpa.hpp"
#include "intel_gpu/op/stateless_kv.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "ov_ops/rms.hpp"

namespace {

bool is_supported_compressed_kv_type(const ov::element::Type& type) {
    return type == ov::element::i8 || type == ov::element::u8 || type == ov::element::i4 || type == ov::element::u4;
}

std::vector<uint64_t> compute_kv_group_sizes(const ov::PartialShape& data_shape, const ov::PartialShape& scale_shape) {
    if (data_shape.rank().is_dynamic()) {
        return {};
    }

    const size_t rank = data_shape.rank().get_length();
    std::vector<uint64_t> group_sizes(rank, 1);
    if (scale_shape.rank().is_static() && static_cast<size_t>(scale_shape.rank().get_length()) == rank) {
        for (size_t i = 0; i < rank; ++i) {
            const bool scale_is_one = scale_shape[i].is_static() && scale_shape[i].get_length() == 1;
            const bool data_is_one = data_shape[i].is_static() && data_shape[i].get_length() == 1;
            if (scale_is_one && !data_is_one) {
                group_sizes[i] = std::numeric_limits<uint64_t>::max();
            }
        }
    } else if (rank > 0) {
        group_sizes[rank - 1] = std::numeric_limits<uint64_t>::max();
    }
    return group_sizes;
}

}  // namespace

namespace ov::intel_gpu {

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v8 = ov::op::v8;


struct GroupQueryAttentionDecomposition::StatelessKVMetadata : public ov::pass::GroupQueryAttentionDecomposition::KVCacheMetadata {
    bool compressed_kv = false;
    bool is_int4_kv = false;
    int64_t kv_num_heads = 0;
    ov::PartialShape past_key_shape;
    std::vector<int64_t> transpose_v_order;
    ~StatelessKVMetadata() override = default;
};

std::unique_ptr<ov::pass::GroupQueryAttentionDecomposition::KVCacheMetadata> GroupQueryAttentionDecomposition::create_metadata() {
    return std::make_unique<StatelessKVMetadata>();
}

ov::Output<ov::Node> GroupQueryAttentionDecomposition::make_qk_norm(const ov::Output<ov::Node>& input, const ov::Output<ov::Node>& weight, float epsilon) {
    return register_new_node<ov::op::internal::RMS>(input, weight, epsilon, input.get_element_type());
}

GroupQueryAttentionDecomposition::KVCacheOutputs GroupQueryAttentionDecomposition::construct_kvcache(
    const std::shared_ptr<ov::op::internal::GroupQueryAttention>& node,
    const ov::Output<ov::Node>& past_key,
    const ov::Output<ov::Node>& past_value,
    const ov::Output<ov::Node>& key,
    const ov::Output<ov::Node>& value,
    const ov::Output<ov::Node>& seqlens_1d,
    const std::shared_ptr<ov::Node>& kv_seqlen,
    const std::shared_ptr<ov::Node>& past_seqlen,
    const ov::Output<ov::Node>& current_seqlen_scalar,
    KVCacheMetadata& metadata_) {
    using namespace ov::op;
    static const std::vector<int64_t> transpose_v_order{0, 1, 3, 2};
    
    auto& metadata = static_cast<StatelessKVMetadata&>(metadata_);
    const auto kv_cache_bit_width = node->get_kv_cache_bit_width();
    metadata.is_int4_kv = kv_cache_bit_width == 4;
    metadata.kv_num_heads = node->get_kv_num_heads();
    metadata.past_key_shape = past_key.get_partial_shape();

    const bool query_seqlens = node->has_query_seqlens();
    // A declared fixed capacity has to grow past its own row count whenever total_sequence_length asks for more
    // headroom (ORT: max(past_rows, total_sequence_length)). Applied to whichever operand feeds StatelessKV,
    // on whichever axis holds the sequence dimension there -- axis 2 normally, axis 3 once transpose_v fusion
    // below is applied (its layout is [B,H,D,S]), since the plugin is production-dependent on both.
    const auto total_seqlen_pos = static_cast<size_t>(ov::op::internal::GroupQueryAttentionInputs::TOTAL_SEQUENCE_LENGTH);
    const bool needs_capacity_growth = query_seqlens && !node->get_sliding_window_cache() && node->has_fixed_kv_capacity() &&
                                       total_seqlen_pos < node->get_input_size() && !ov::util::is_empty_constant_tensor(node->input_value(total_seqlen_pos));
    const bool need_qk_output = node->get_qk_output() != ov::op::internal::GroupQueryAttentionQKOutput::NONE;
    const bool use_explicit_attention = node->get_softcap() > 0.0F || need_qk_output;
    // The explicit attention chain consumes plain (dequantized) K/V, so skip the compressed-KV SDPA form.
    if (!use_explicit_attention) {
        // covers ExpandBroadcastReshapeSDPAFusion, gpu sdpa supports broadcast natively
        metadata.should_broadcast_kv = false;
        using GQAQuantType = ov::op::internal::GroupQueryAttentionQuantType;
        const auto& key_type = past_key.get_element_type();
        // Per-channel compressed KV requires the SDPA micro kernel.
        metadata.compressed_kv = node->is_kv_quantized() && (kv_cache_bit_width == 8 || metadata.is_int4_kv) && key_type == past_value.get_element_type() &&
                                 is_supported_compressed_kv_type(key_type) && node->get_k_quant_type() == GQAQuantType::PER_CHANNEL &&
                                 node->get_v_quant_type() == GQAQuantType::PER_CHANNEL && m_sdpa_support_immad;
    }

    const auto get_exclusive_to_result = [](const Output<ov::Node>& output) -> const v0::Result* {
        const auto inputs = output.get_target_inputs();
        if (inputs.size() != 1) {
            GPU_DEBUG_TRACE_DETAIL << "GQA's output " << output.get_index() << " connects to " << inputs.size() << " nodes" << std::endl;
            return nullptr;
        }
        ov::Node* const node = inputs.begin()->get_node();
        if (!ov::is_type<v0::Result>(node)) {
            GPU_DEBUG_TRACE_DETAIL << "GQA's output " << output.get_index() << " not connects to Result, get: " << node->get_type_name() << ":"
                                   << node->get_friendly_name() << std::endl;
            return nullptr;
        }
        return ov::as_type<v0::Result>(node);
    };
    const v0::Result* present_key = get_exclusive_to_result(node->output(1));
    const v0::Result* present_value = get_exclusive_to_result(node->output(2));

    KVCacheOutputs outputs;
    const bool sliding_window = node->get_sliding_window_cache();
    const int64_t local_window_size = node->get_local_window_size();
    std::optional<ov::Output<ov::Node>> cur_seq_len;
    if (query_seqlens) {
        cur_seq_len = node->input_value(static_cast<size_t>(ov::op::internal::GroupQueryAttentionInputs::QUERY_SEQLENS));
    }
    const auto make_stateless_kv = [&](const ov::Output<ov::Node>& past, const ov::Output<ov::Node>& current, int64_t concat_axis) {
        if (query_seqlens) {
            return register_new_node<op::StatelessKV>(past, current, seqlens_1d, *cur_seq_len, concat_axis, true, local_window_size, sliding_window);
        }
        return register_new_node<op::StatelessKV>(past, current, seqlens_1d, concat_axis, true, local_window_size, sliding_window);
    };

    const auto value_transpose_order = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{4}, transpose_v_order));
    // Identifies the node itself, not a specific output/input port, so it takes a node rather than an Output.
    const auto find_value_transpose = [](const std::shared_ptr<ov::Node>& candidate) -> std::shared_ptr<v1::Transpose> {
        const auto transpose = ov::as_type_ptr<v1::Transpose>(candidate);
        if (!transpose) {
            return {};
        }
        const auto order = ov::as_type_ptr<v0::Constant>(transpose->input_value(1).get_node_shared_ptr());
        if (!order || order->cast_vector<int64_t>() != transpose_v_order) {
            return {};
        }
        return transpose;
    };

    std::shared_ptr<v1::Transpose> past_value_transpose;
    std::shared_ptr<v1::Transpose> present_value_transpose;
    // explicit attention hard-coded layout so don't fuse
    if (!use_explicit_attention) {
        past_value_transpose = find_value_transpose(node->input_value(4).get_node_shared_ptr());
        if (node->output(2).get_target_inputs().size() == 1) {
            const auto target = *node->output(2).get_target_inputs().begin();
            present_value_transpose = find_value_transpose(target.get_node()->shared_from_this());
        }
    }
    const bool transpose_value_cache = past_value_transpose && present_value_transpose;
    if (transpose_value_cache) {
        OPENVINO_ASSERT(!present_value);
        GPU_DEBUG_TRACE_DETAIL << "pending transpose_v fusion on " << present_value_transpose->get_friendly_name() << std::endl;
        present_value = get_exclusive_to_result(present_value_transpose->output(0));
    }

    if (!present_key || !present_value) {
        GPU_DEBUG_TRACE_DETAIL << "GQA not having key&value both exclusively connects to Result, key: "
                               << (present_key ? present_key->get_friendly_name() : "null")
                               << " value: " << (present_value ? present_value->get_friendly_name() : "null") << " so fallback to common decompose"
                               << std::endl;
        auto outputs = ov::pass::GroupQueryAttentionDecomposition::construct_kvcache(node,
                                                                                     past_key,
                                                                                     past_value,
                                                                                     key,
                                                                                     value,
                                                                                     seqlens_1d,
                                                                                     kv_seqlen,
                                                                                     past_seqlen,
                                                                                     current_seqlen_scalar,
                                                                                     metadata_);
        if (!use_explicit_attention && node->get_causal() && !node->get_sliding_window_cache() && node->has_fixed_kv_capacity()) {
            // static path put full buffer to SDPA, trim it for causal.
            const auto slice_start = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}));
            const auto slice_step = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}));
            const auto slice_axis = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}));
            const auto sdpa_end = register_new_node<v1::Add>(past_seqlen, current_seqlen_scalar);
            outputs.sdpa_key = register_new_node<v8::Slice>(outputs.sdpa_key, slice_start, sdpa_end, slice_step, slice_axis);
            outputs.sdpa_value = register_new_node<v8::Slice>(outputs.sdpa_value, slice_start, sdpa_end, slice_step, slice_axis);
            GPU_DEBUG_TRACE_DETAIL << "GQA static path fix: SDPA key/value trimmed" << std::endl;
        }
        metadata.should_dequantize_kv = metadata.should_dequantize_kv && !metadata.compressed_kv;
        return outputs;
    }

    // resize_kv_to_capacity is a no-op (in value, not necessarily in graph nodes -- see the function's own
    // early-return comment) whenever capacity already covers the request, so StatelessKV stays zero-copy for
    // every already-covered config regardless of transpose_v fusion. past_key and an untransposed past_value
    // share the same sequence-dim value, so past_rows (axis 2) is computed once and reused for both; the
    // transposed value layout needs its own axis-3 read since it's a different tensor on a different axis.
    ov::Output<ov::Node> total_sequence_length;
    ov::Output<ov::Node> past_rows;
    if (needs_capacity_growth) {
        const auto one = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}));
        total_sequence_length =
            register_new_node<v1::Reshape>(register_new_node<v0::Convert>(node->input_value(total_seqlen_pos), ov::element::i64), one, false);
        past_rows = get_dimensions(register_new_node<v3::ShapeOf>(past_key), {2});
    }
    ov::Output<ov::Node> stateless_past_key = past_key;
    if (needs_capacity_growth) {
        stateless_past_key = resize_kv_to_capacity(past_rows, past_key, total_sequence_length);
    }

    const auto key_cache = make_stateless_kv(stateless_past_key, key, 2);
    key_cache->set_friendly_name(present_key->get_friendly_name() + "_stateless");

    std::shared_ptr<op::StatelessKV> value_cache;
    if (transpose_value_cache) {
        if (metadata.is_int4_kv) {
            metadata.compressed_kv = false;
        }
        // Before: past_value -> Transpose -> GQA cache -> Transpose -> present_value.
        // After:  past_value -------------------------> StatelessKV(axis=3) -> present_value.
        //         current_value -> Transpose ----------^      (the output Transpose is bypassed)
        ov::Output<ov::Node> stateless_past_value_transposed = past_value_transpose->input_value(0);
        if (needs_capacity_growth) {
            // The transposed buffer's sequence dimension is axis 3 ([B,H,D,S]), not axis 2.
            const auto transposed_past_rows = get_dimensions(register_new_node<v3::ShapeOf>(stateless_past_value_transposed), {3});
            stateless_past_value_transposed = resize_kv_to_capacity(transposed_past_rows, stateless_past_value_transposed, total_sequence_length, 3);
        }
        const auto transposed_value = register_new_node<v1::Transpose>(value, value_transpose_order);
        value_cache = make_stateless_kv(stateless_past_value_transposed, transposed_value, 3);
        present_value_transpose->output(0).replace(value_cache->output(0));
    } else {
        ov::Output<ov::Node> stateless_past_value = past_value;
        if (needs_capacity_growth) {
            stateless_past_value = resize_kv_to_capacity(past_rows, past_value, total_sequence_length);
        }
        value_cache = make_stateless_kv(stateless_past_value, value, 2);
    }
    value_cache->set_friendly_name(present_value->get_friendly_name() + "_stateless");

    outputs.present_key = key_cache->output(0);
    outputs.present_value = value_cache->output(0);
    if (need_qk_output) {
        // need_qk_output feeds the full (untrimmed) present buffer straight into explicit attention
        // (see make_explicit_attention), unlike the output(1) SDPA view below, which StatelessKV itself
        // already trims to the written length. Positions >= the valid total (seqlens_1d) may hold
        // never-written filler there: zero from resize_kv_to_capacity's growth, or stale/uninitialized
        // rows within the original declared capacity that StatelessKV never writes. An additive attention
        // mask can't neutralize NaN in that filler (NaN + mask bias is still NaN), so zero it directly on
        // this copy only -- present_key/value above must stay wired straight to StatelessKV's own output,
        // required by the plugin's stateless_kv-to-output aliasing (primitive_inst::
        // realloc_outputs_for_stateless_kv asserts on it). Mirrors the common decomposition's identical
        // has_fixed_kv_capacity() + query_seqlens safeguard (construct_kvcache, is_static_input branch,
        // this transformation's common/CPU base class).
        const auto zero_out_unwritten_tail = [&](const ov::Output<ov::Node>& buffer, int64_t axis) {
            const auto capacity = get_dimensions(register_new_node<v3::ShapeOf>(buffer), {static_cast<int>(axis)});
            std::vector<int64_t> index_shape(4, 1);
            index_shape[axis] = -1;
            const auto index_shape_node = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{4}, index_shape));
            const auto zero_scalar = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
            const auto one_scalar = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{}, {1}));
            const auto rows = register_new_node<v1::Reshape>(
                register_new_node<v4::Range>(zero_scalar, register_new_node<v0::Squeeze>(capacity), one_scalar, ov::element::i64),
                index_shape_node,
                false);
            const auto valid_cols = register_new_node<v1::Less>(rows, seqlens_1d);
            const auto zero_fill = register_new_node(v0::Constant::create(buffer.get_element_type(), ov::Shape{}, {0}));
            return register_new_node<v1::Select>(valid_cols, buffer, zero_fill);
        };
        outputs.sdpa_key = needs_capacity_growth ? zero_out_unwritten_tail(outputs.present_key, 2) : outputs.present_key;
        outputs.sdpa_value = needs_capacity_growth ? zero_out_unwritten_tail(outputs.present_value, transpose_value_cache ? 3 : 2) : outputs.present_value;
    } else {
        outputs.sdpa_key = key_cache->output(1);
        outputs.sdpa_value = value_cache->output(1);
    }

    const auto zero = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}));
    if (sliding_window) {
        OPENVINO_ASSERT(!need_qk_output, "[GPU] SWA does not support qk output");
    }
    if (local_window_size > 0 && !need_qk_output) {
        // Must match StatelessKV's sdpa_front = max(0, P - window + 1).
        const auto zero_scalar = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
        const auto window_minus_one = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{}, {local_window_size - 1}));
        const auto past_scalar = register_new_node<v0::Squeeze>(past_seqlen);
        const auto sdpa_front = register_new_node<v1::Maximum>(register_new_node<v1::Subtract>(past_scalar, window_minus_one), zero_scalar);
        const auto resident_past = register_new_node<v1::Subtract>(past_scalar, sdpa_front);
        outputs.mask_past_seqlen = register_new_node<v0::Unsqueeze>(resident_past, zero);
        outputs.bias_col_offset = register_new_node<v0::Unsqueeze>(sdpa_front, zero);
    } else {
        outputs.mask_past_seqlen = past_seqlen;
        outputs.bias_col_offset = zero;
    }

    metadata.should_dequantize_kv = metadata.should_dequantize_kv && !metadata.compressed_kv;
    if (transpose_value_cache) {
        if (metadata.should_dequantize_kv) {
            outputs.sdpa_value = register_new_node<v1::Transpose>(outputs.sdpa_value, value_transpose_order);
        } else {
            metadata.transpose_v_order = transpose_v_order;
        }
    }
    return outputs;
}

std::shared_ptr<ov::Node> GroupQueryAttentionDecomposition::make_sdpa(const ov::Output<ov::Node>& query,
                                                                      const ov::Output<ov::Node>& key,
                                                                      const ov::Output<ov::Node>& value,
                                                                      const ov::Output<ov::Node>& mask,
                                                                      const ov::Output<ov::Node>& scale,
                                                                      const ov::Output<ov::Node>& sink,
                                                                      bool is_causal,
                                                                      int64_t local_window_size,
                                                                      const KVCacheMetadata& metadata_) {
    const auto& metadata = static_cast<const StatelessKVMetadata&>(metadata_);
    ov::OutputVector inputs{query, key, value};
    if (mask.get_node()) {
        inputs.push_back(mask);
    }
    if (scale.get_node()) {
        inputs.push_back(scale);
    }
    if (sink.get_node()) {
        inputs.push_back(sink);
    }

    const auto order = op::SDPA::default_order(query.get_partial_shape().rank().get_length());
    const auto& value_order = metadata.transpose_v_order.empty() ? order : metadata.transpose_v_order;
    const auto alignment = is_causal ? op::SDPA::CausalMaskAlignment::LOWER_RIGHT : op::SDPA::CausalMaskAlignment::UPPER_LEFT;
    // GQA's -1 sentinel for "no window" maps to SDPA's 0 = disabled.
    const int64_t sdpa_window = (local_window_size >= 1) ? local_window_size : 0;
    std::shared_ptr<op::SDPA> sdpa;
    if (metadata.compressed_kv) {
        using GQAQuantType = ov::op::internal::GroupQueryAttentionQuantType;
        using DynamicQuantize = ov::op::internal::DynamicQuantize;
        const bool is_int4 = metadata.is_int4_kv;
        OPENVINO_ASSERT(!is_int4 || value_order == order, "Compressed INT4 GQA SDPA requires a non-transposed V input");

        const auto prepare_scale = [&](const ov::Output<ov::Node>& scale_input) {
            ov::Output<ov::Node> prepared_scale = make_kv_scale(scale_input, metadata.kv_num_heads, GQAQuantType::PER_CHANNEL);
            if (prepared_scale.get_element_type() != ov::element::f16) {
                prepared_scale = register_new_node<v0::Convert>(prepared_scale, ov::element::f16);
            }
            return prepared_scale;
        };
        const auto prepared_key_scale = prepare_scale(metadata.k_scale);
        const auto prepared_value_scale = prepare_scale(metadata.v_scale);
        inputs.push_back(prepared_key_scale);
        inputs.push_back(prepared_value_scale);

        op::SDPA::QuantizationAttribute quantization_attrs;
        quantization_attrs.quantization_type = is_int4 ? DynamicQuantize::QuantizationType::Asymmetric : DynamicQuantize::QuantizationType::Symmetric;
        quantization_attrs.output_storage_type = DynamicQuantize::OutputStorageType::Planar;
        // GQA stores signed INT4 values as unsigned nibbles biased by +8. Describe that physical encoding to SDPA
        // as asymmetric u4 with a fixed zero point instead of i4, whose nibbles use two's-complement encoding.
        quantization_attrs.quantization_dt = is_int4 ? ov::element::u4 : key.get_element_type();
        quantization_attrs.scale_dt = ov::element::f16;
        if (is_int4) {
            quantization_attrs.zp_dt = ov::element::f16;
            const auto storage_zp = register_new_node<v0::Constant>(ov::element::f16, ov::Shape{}, 8.0F);
            const auto key_scale_shape = register_new_node<ov::op::v3::ShapeOf>(prepared_key_scale, ov::element::i64);
            const auto value_scale_shape = register_new_node<ov::op::v3::ShapeOf>(prepared_value_scale, ov::element::i64);
            inputs.push_back(register_new_node<ov::op::v3::Broadcast>(storage_zp, key_scale_shape));
            inputs.push_back(register_new_node<ov::op::v3::Broadcast>(storage_zp, value_scale_shape));
        }
        quantization_attrs.group_sizes = compute_kv_group_sizes(metadata.past_key_shape, prepared_key_scale.get_partial_shape());
        quantization_attrs.scales_zp_output_order = {0, 1, 2, 3};

        sdpa =
            register_new_node<op::SDPA>(inputs, is_causal, order, order, value_order, order, quantization_attrs, ov::element::dynamic, alignment, sdpa_window);
    } else {
        sdpa = register_new_node<op::SDPA>(inputs, is_causal, order, order, value_order, order, ov::element::dynamic, alignment, sdpa_window);
    }
    return sdpa;
}

std::shared_ptr<ov::Node> GroupQueryAttentionDecomposition::make_attention_mask(const ov::Output<ov::Node>& curr_seqlen_scalar,
                                                                                const ov::Output<ov::Node>& kv_len_scalar,
                                                                                const ov::Output<ov::Node>& kv_len_1d,
                                                                                const ov::Output<ov::Node>& past_seqlen,
                                                                                const ov::element::Type& compute_type,
                                                                                bool causal,
                                                                                int64_t local_window_size,
                                                                                const ov::Output<ov::Node>& external_bias,
                                                                                const ov::Output<ov::Node>& bias_col_offset,
                                                                                bool sliding_window_cache,
                                                                                float scale,
                                                                                bool has_sink,
                                                                                [[maybe_unused]] const KVCacheMetadata& metadata_) {
    // The kernel applies the window natively via is_causal + sliding_window_size (see make_sdpa),
    // so the explicit mask subgraph is unneeded whenever that path is reachable.
    if (causal && !external_bias.get_node() && scale == 0.0F && !has_sink) {
        return nullptr;
    }

    return ov::pass::GroupQueryAttentionDecomposition::make_attention_mask(curr_seqlen_scalar,
                                                                           kv_len_scalar,
                                                                           kv_len_1d,
                                                                           past_seqlen,
                                                                           compute_type,
                                                                           causal,
                                                                           local_window_size,
                                                                           external_bias,
                                                                           bias_col_offset,
                                                                           sliding_window_cache,
                                                                           scale,
                                                                           has_sink,
                                                                           metadata_);
}

}  // namespace ov::intel_gpu
