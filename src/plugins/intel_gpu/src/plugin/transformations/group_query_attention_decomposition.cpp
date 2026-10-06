// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "group_query_attention_decomposition.hpp"

#include <limits>
#include <utility>

#include "intel_gpu/op/sdpa.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/shape_of.hpp"

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

std::optional<GroupQueryAttentionDecomposition::CompressedKV> GroupQueryAttentionDecomposition::prepare_compressed_kv(
    const std::shared_ptr<ov::op::internal::GroupQueryAttention>& node,
    const ov::Output<ov::Node>& key,
    const ov::Output<ov::Node>& value,
    const ov::Output<ov::Node>& key_scale,
    const ov::Output<ov::Node>& value_scale) {
    using GQAInputs = ov::op::internal::GroupQueryAttentionInputs;
    using GQAQuantType = ov::op::internal::GroupQueryAttentionQuantType;

    const auto kv_cache_bit_width = node->get_kv_cache_bit_width();
    const auto key_quant_type = node->get_k_quant_type();
    const auto value_quant_type = node->get_v_quant_type();
    if (!node->is_kv_quantized() || (kv_cache_bit_width != 8 && kv_cache_bit_width != 4) || key.get_element_type() != value.get_element_type() ||
        !is_supported_compressed_kv_type(key.get_element_type()) || key_quant_type == GQAQuantType::PER_TENSOR ||
        value_quant_type == GQAQuantType::PER_TENSOR)
        return std::nullopt;
    // Per-channel compressed KV is only handled by the micro SDPA kernel, which has no f32 KQ gemm: an f32 compute type
    // (f32 inference precision) would fall back to kernels that read the scales per token. Dequantize explicitly instead.
    if (node->get_input_element_type(static_cast<size_t>(GQAInputs::QUERY)) == ov::element::f32) {
        return std::nullopt;
    }

    ov::Output<ov::Node> prepared_key_scale =
        make_kv_scale(node->input_value(static_cast<size_t>(GQAInputs::K_SCALE)), node->get_kv_num_heads(), key_quant_type);
    ov::Output<ov::Node> prepared_value_scale =
        make_kv_scale(node->input_value(static_cast<size_t>(GQAInputs::V_SCALE)), node->get_kv_num_heads(), value_quant_type);

    if (prepared_key_scale.get_element_type() != ov::element::f16)
        prepared_key_scale = register_new_node<ov::op::v0::Convert>(prepared_key_scale, ov::element::f16);
    if (prepared_value_scale.get_element_type() != ov::element::f16)
        prepared_value_scale = register_new_node<ov::op::v0::Convert>(prepared_value_scale, ov::element::f16);

    const bool is_int4 = kv_cache_bit_width == 4;
    m_quantization_attrs.quantization_type = is_int4 ? ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric
                                                     : ov::op::internal::DynamicQuantize::QuantizationType::Symmetric;
    m_quantization_attrs.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::Planar;
    // GQA stores signed INT4 values as unsigned nibbles biased by +8. Describe that physical encoding to SDPA
    // as asymmetric u4 with a fixed zero point instead of i4, whose nibbles use two's-complement encoding.
    m_quantization_attrs.quantization_dt = is_int4 ? ov::element::u4 : key.get_element_type();
    m_quantization_attrs.scale_dt = ov::element::f16;
    ov::OutputVector quantization_inputs{prepared_key_scale, prepared_value_scale};
    if (is_int4) {
        m_quantization_attrs.zp_dt = ov::element::f16;
        const auto storage_zp = register_new_node<ov::op::v0::Constant>(ov::element::f16, ov::Shape{}, 8.0f);
        const auto key_scale_shape = register_new_node<ov::op::v3::ShapeOf>(prepared_key_scale, ov::element::i64);
        const auto value_scale_shape = register_new_node<ov::op::v3::ShapeOf>(prepared_value_scale, ov::element::i64);
        quantization_inputs.push_back(register_new_node<ov::op::v3::Broadcast>(storage_zp, key_scale_shape));
        quantization_inputs.push_back(register_new_node<ov::op::v3::Broadcast>(storage_zp, value_scale_shape));
    }
    m_quantization_attrs.group_sizes = compute_kv_group_sizes(key.get_partial_shape(), prepared_key_scale.get_partial_shape());
    m_quantization_attrs.scales_zp_output_order = {0, 1, 2, 3};
    return CompressedKV{key, value, std::move(quantization_inputs)};
}

std::shared_ptr<ov::Node> GroupQueryAttentionDecomposition::make_sdpa(const ov::Output<ov::Node>& query,
                                                                      const ov::Output<ov::Node>& key,
                                                                      const ov::Output<ov::Node>& value,
                                                                      const ov::Output<ov::Node>& mask,
                                                                      const ov::Output<ov::Node>& scale,
                                                                      const ov::Output<ov::Node>& sink,
                                                                      bool is_causal,
                                                                      const std::optional<CompressedKV>& compressed_kv) {
    ov::OutputVector inputs{query,
                            compressed_kv ? compressed_kv->key : key,
                            compressed_kv ? compressed_kv->value : value};
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
        inputs.insert(inputs.end(),
                      compressed_kv->quantization_inputs.begin(),
                      compressed_kv->quantization_inputs.end());
    }

    const auto order = op::SDPA::default_order(query.get_partial_shape().rank().get_length());
    const auto alignment = is_causal ? op::SDPA::CausalMaskAlignment::LOWER_RIGHT : op::SDPA::CausalMaskAlignment::UPPER_LEFT;
    std::shared_ptr<op::SDPA> sdpa;
    if (compressed_kv) {
        sdpa = register_new_node<op::SDPA>(inputs, is_causal, order, order, order, order, m_quantization_attrs, ov::element::dynamic, alignment);
    } else {
        sdpa = register_new_node<op::SDPA>(inputs, is_causal, order, order, order, order, ov::element::dynamic, alignment);
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
                                                                                bool has_sink) {
    // Mask-free LOWER_RIGHT causal SDPA is only valid when the KV length equals past + current (Concat path, dynamic
    // length). A statically-sized KV buffer (full-length static cache) has unused tail slots that must stay masked.
    const bool kv_len_static = ov::util::get_constant_from_source(kv_len_1d) != nullptr;
    if (causal && local_window_size == -1 && !sliding_window_cache && !external_bias.get_node() && scale == 0.0f && !has_sink && !kv_len_static) {
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
                                                                           has_sink);
}

}  // namespace ov::intel_gpu
