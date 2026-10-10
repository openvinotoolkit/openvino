// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>
#include <optional>
#include <utility>

#include "group_query_attention_decomposition.hpp"

#include "intel_gpu/op/sdpa.hpp"
#include "intel_gpu/op/stateless_kv.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/transpose.hpp"

namespace {

bool is_supported_compressed_kv_type(const ov::element::Type& type) {
    return type == ov::element::i8 || type == ov::element::u8 || type == ov::element::i4 || type == ov::element::u4;
}

std::vector<uint64_t> compute_kv_group_sizes(const ov::PartialShape& data_shape, const ov::PartialShape& scale_shape) {
    if (data_shape.rank().is_dynamic())
        return {};

    const size_t rank = data_shape.rank().get_length();
    std::vector<uint64_t> group_sizes(rank, 1);
    if (scale_shape.rank().is_static() && static_cast<size_t>(scale_shape.rank().get_length()) == rank) {
        for (size_t i = 0; i < rank; ++i) {
            const bool scale_is_one = scale_shape[i].is_static() && scale_shape[i].get_length() == 1;
            const bool data_is_one = data_shape[i].is_static() && data_shape[i].get_length() == 1;
            if (scale_is_one && !data_is_one)
                group_sizes[i] = std::numeric_limits<uint64_t>::max();
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


struct GroupQueryAttentionDecomposition::StatelessKVMetadata : public ov::pass::GroupQueryAttentionDecomposition::KVCacheMetadata {
    struct CompressedKV {
        ov::Output<ov::Node> key;
        ov::Output<ov::Node> value;
        ov::OutputVector quantization_inputs;
        op::SDPA::QuantizationAttribute quantization_attrs;
    };
    std::optional<CompressedKV> compressed_kv;
    std::vector<int64_t> transpose_v_order;
    ~StatelessKVMetadata() override = default;
};

void GroupQueryAttentionDecomposition::prepare_compressed_kv(const std::shared_ptr<ov::op::internal::GroupQueryAttention>& node,
                                                             const ov::Output<ov::Node>& key,
                                                             const ov::Output<ov::Node>& value,
                                                             KVCacheMetadata& metadata_) {
    using GQAInputs = ov::op::internal::GroupQueryAttentionInputs;
    using GQAQuantType = ov::op::internal::GroupQueryAttentionQuantType;

    auto& metadata = static_cast<StatelessKVMetadata&>(metadata_);
    // covers ExpandBroadcastReshapeSDPAFusion, gpu sdpa supports broadcast natively
    metadata.should_broadcast_kv = false; 

    const auto kv_cache_bit_width = node->get_kv_cache_bit_width();
    const auto key_quant_type = node->get_k_quant_type();
    const auto value_quant_type = node->get_v_quant_type();
    if (!node->is_kv_quantized() || (kv_cache_bit_width != 8 && kv_cache_bit_width != 4) || key.get_element_type() != value.get_element_type() ||
        !is_supported_compressed_kv_type(key.get_element_type()) || key_quant_type == GQAQuantType::PER_TENSOR || value_quant_type == GQAQuantType::PER_TENSOR)
        return;

    auto& compressed_kv = metadata.compressed_kv.emplace();
    compressed_kv.key = key;
    compressed_kv.value = value;
    
    ov::Output<ov::Node> prepared_key_scale =
        make_kv_scale(node->input_value(static_cast<size_t>(GQAInputs::K_SCALE)), node->get_kv_num_heads(), key_quant_type);
    ov::Output<ov::Node> prepared_value_scale =
        make_kv_scale(node->input_value(static_cast<size_t>(GQAInputs::V_SCALE)), node->get_kv_num_heads(), value_quant_type);

    if (prepared_key_scale.get_element_type() != ov::element::f16)
        prepared_key_scale = register_new_node<ov::op::v0::Convert>(prepared_key_scale, ov::element::f16);
    if (prepared_value_scale.get_element_type() != ov::element::f16)
        prepared_value_scale = register_new_node<ov::op::v0::Convert>(prepared_value_scale, ov::element::f16);

    const bool is_int4 = kv_cache_bit_width == 4;
    compressed_kv.quantization_attrs.quantization_type =
        is_int4 ? ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric : ov::op::internal::DynamicQuantize::QuantizationType::Symmetric;
    compressed_kv.quantization_attrs.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::Planar;
    // GQA stores signed INT4 values as unsigned nibbles biased by +8. Describe that physical encoding to SDPA
    // as asymmetric u4 with a fixed zero point instead of i4, whose nibbles use two's-complement encoding.
    compressed_kv.quantization_attrs.quantization_dt = is_int4 ? ov::element::u4 : key.get_element_type();
    compressed_kv.quantization_attrs.scale_dt = ov::element::f16;
    compressed_kv.quantization_inputs = {prepared_key_scale, prepared_value_scale};
    if (is_int4) {
        compressed_kv.quantization_attrs.zp_dt = ov::element::f16;
        const auto storage_zp = register_new_node<ov::op::v0::Constant>(ov::element::f16, ov::Shape{}, 8.0f);
        const auto key_scale_shape = register_new_node<ov::op::v3::ShapeOf>(prepared_key_scale, ov::element::i64);
        const auto value_scale_shape = register_new_node<ov::op::v3::ShapeOf>(prepared_value_scale, ov::element::i64);
        compressed_kv.quantization_inputs.push_back(register_new_node<ov::op::v3::Broadcast>(storage_zp, key_scale_shape));
        compressed_kv.quantization_inputs.push_back(register_new_node<ov::op::v3::Broadcast>(storage_zp, value_scale_shape));
    }
    compressed_kv.quantization_attrs.group_sizes = compute_kv_group_sizes(key.get_partial_shape(), prepared_key_scale.get_partial_shape());
    compressed_kv.quantization_attrs.scales_zp_output_order = {0, 1, 2, 3};

    metadata.should_dequantize_kv = false;
}

std::unique_ptr<ov::pass::GroupQueryAttentionDecomposition::KVCacheMetadata> GroupQueryAttentionDecomposition::create_metadata(
    const std::shared_ptr<ov::op::internal::GroupQueryAttention>& node) {
    return std::make_unique<StatelessKVMetadata>();
}

GroupQueryAttentionDecomposition::KVCacheOutputs GroupQueryAttentionDecomposition::construct_kvcache(
    const std::shared_ptr<ov::op::internal::GroupQueryAttention>& node,
    const ov::Output<ov::Node>& past_key,
    const ov::Output<ov::Node>& past_value,
    const ov::Output<ov::Node>& key,
    const ov::Output<ov::Node>& value,
    const ov::Output<ov::Node>& seqlens_1d,
    const ov::Output<ov::Node>& past_seqlen,
    const ov::Output<ov::Node>& current_seqlen_scalar,
    KVCacheMetadata& metadata_) {
    using namespace ov::op;
    static const std::vector<int64_t> transpose_v_order{0, 1, 3, 2};
    
    auto& metadata = static_cast<StatelessKVMetadata&>(metadata_);

    KVCacheOutputs outputs;

    if (!node->get_sliding_window_cache()) {

        const auto key_cache = register_new_node<op::StatelessKV>(past_key, key, seqlens_1d, 2, true);

        const auto value_transpose_order = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{4}, transpose_v_order));
        const auto find_value_transpose = [](const ov::Output<ov::Node>& output) {
            const auto transpose = ov::as_type_ptr<v1::Transpose>(output.get_node_shared_ptr());
            if (!transpose) {
                return std::shared_ptr<v1::Transpose>{};
            }
            const auto order = ov::as_type_ptr<v0::Constant>(transpose->input_value(1).get_node_shared_ptr());
            if (!order || order->cast_vector<int64_t>() != transpose_v_order) {
                return std::shared_ptr<v1::Transpose>{};
            }
            return transpose;
        };

        const auto input_past_value = node->input_value(4);
        const auto past_value_transpose = find_value_transpose(input_past_value);
        std::shared_ptr<v1::Transpose> present_value_transpose;
        if (node->output(2).get_target_inputs().size() == 1) {
            const auto target = *node->output(2).get_target_inputs().begin();
            present_value_transpose = find_value_transpose(target.get_node()->output(target.get_index()));
        }

        const auto transpose_v = past_value_transpose && present_value_transpose;
        std::shared_ptr<op::StatelessKV> value_cache;
        if (transpose_v) {
            metadata.transpose_v_order = transpose_v_order;
            // Before: past_value -> Transpose -> GQA cache -> Transpose -> present_value.
            // After:  past_value -------------------------> StatelessKV(axis=3) -> present_value.
            //         current_value -> Transpose ----------^      (the output Transpose is bypassed)
            const auto transposed_value = register_new_node<v1::Transpose>(value, value_transpose_order);
            value_cache = register_new_node<op::StatelessKV>(past_value_transpose->input_value(0), transposed_value, seqlens_1d, 3, true);
            present_value_transpose->output(0).replace(value_cache->output(0));
        } else {
            value_cache = register_new_node<op::StatelessKV>(past_value, value, seqlens_1d, 2, true);
        }

        outputs.present_key = key_cache->output(0);
        outputs.present_value = value_cache->output(0);
        outputs.sdpa_key = key_cache->output(1);
        outputs.sdpa_value = value_cache->output(1);
        outputs.mask_past_seqlen = past_seqlen;
        outputs.bias_col_offset = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}));

    } else {
        outputs = ov::pass::GroupQueryAttentionDecomposition::construct_kvcache(node,
                                                                                past_key,
                                                                                past_value,
                                                                                key,
                                                                                value,
                                                                                seqlens_1d,
                                                                                past_seqlen,
                                                                                current_seqlen_scalar,
                                                                                metadata);
    }

    prepare_compressed_kv(node, outputs.sdpa_key, outputs.sdpa_value, metadata);
    return outputs;
}

std::shared_ptr<ov::Node> GroupQueryAttentionDecomposition::make_sdpa(const ov::Output<ov::Node>& query,
                                                                      const ov::Output<ov::Node>& key,
                                                                      const ov::Output<ov::Node>& value,
                                                                      const ov::Output<ov::Node>& mask,
                                                                      const ov::Output<ov::Node>& scale,
                                                                      const ov::Output<ov::Node>& sink,
                                                                      bool is_causal,
                                                                      const KVCacheMetadata& metadata_) {
    const auto& metadata = static_cast<const StatelessKVMetadata&>(metadata_);
    const auto& compressed_kv = metadata.compressed_kv;
    ov::OutputVector inputs{query, compressed_kv ? compressed_kv->key : key, compressed_kv ? compressed_kv->value : value};
    if (mask.get_node()) {
        inputs.push_back(mask);
    }
    if (scale.get_node()) {
        inputs.push_back(scale);
    }
    if (sink.get_node()) {
        inputs.push_back(sink);
    }
    if (compressed_kv) {
        inputs.insert(inputs.end(), compressed_kv->quantization_inputs.begin(), compressed_kv->quantization_inputs.end());
    }

    const auto order = op::SDPA::default_order(query.get_partial_shape().rank().get_length());
    const auto& value_order = metadata.transpose_v_order.empty() ? order : metadata.transpose_v_order;
    const auto alignment = is_causal ? op::SDPA::CausalMaskAlignment::LOWER_RIGHT : op::SDPA::CausalMaskAlignment::UPPER_LEFT;
    std::shared_ptr<op::SDPA> sdpa;
    if (compressed_kv) {
        sdpa = register_new_node<op::SDPA>(inputs,
                                           is_causal,
                                           order,
                                           order,
                                           value_order,
                                           order,
                                           compressed_kv->quantization_attrs,
                                           ov::element::dynamic,
                                           alignment);
    } else {
        sdpa = register_new_node<op::SDPA>(inputs, is_causal, order, order, value_order, order, ov::element::dynamic, alignment);
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
    if (causal && local_window_size == -1 && !sliding_window_cache && !external_bias.get_node() && scale == 0.0f && !has_sink) {
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
