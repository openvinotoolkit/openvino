// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kv_cache_compression.hpp"

#include "intel_gpu/op/kv_cache.hpp"
#include "intel_gpu/op/kv_cache_compressed.hpp"
#include "intel_gpu/op/indirect_sdpa.hpp"
#include "intel_gpu/op/read_value.hpp"
#include "intel_gpu/op/read_values.hpp"
#include "intel_gpu/op/sparse_sdpa.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "ov_ops/dynamic_quantize.hpp"

#include "openvino/core/node_vector.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/sink.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "openvino/util/log.hpp"
#include "transformations/utils/utils.hpp"
#include "plugin/transformations/utils.hpp"

#include <memory>
#include "openvino/core/graph_util.hpp"


namespace ov::intel_gpu {

namespace {
std::vector<ov::op::util::VariableInfo> get_variable_infos(const ov::op::util::VariableInfo& data_variable_info,
                                                           const ov::op::internal::DynamicQuantize::Attributes& quantization_attrs) {
    std::vector<ov::op::util::VariableInfo> infos;

    // Add initial data variable info
    infos.push_back(data_variable_info);

    // Infer DQ shapes
    ov::op::internal::DynamicQuantize dq;
    dq.set_attrs(quantization_attrs);

    auto dq_shapes = ov::op::internal::DynamicQuantize::shape_infer(&dq, {data_variable_info.data_shape});

    const auto variable_id = data_variable_info.variable_id;
    const auto scale_shape = dq_shapes[1];
    const auto scale_dt = quantization_attrs.scale_dt;

    // Add scales variable info
    infos.push_back(ov::op::util::VariableInfo{scale_shape, scale_dt, variable_id});

    if (quantization_attrs.quantization_type == ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric &&
        quantization_attrs.output_storage_type == ov::op::internal::DynamicQuantize::OutputStorageType::Planar) {
        // Add zero points variable info
        const auto zp_dt = quantization_attrs.zp_dt;
        infos.push_back(ov::op::util::VariableInfo{scale_shape, zp_dt, variable_id});
    }

    return infos;
}

std::shared_ptr<ov::intel_gpu::op::ReadValues>
    update_past_read_value(std::shared_ptr<ov::intel_gpu::op::ReadValue> past_rv_node,
                           const ov::op::internal::DynamicQuantize::Attributes& quantization_attrs) {
    auto variable = past_rv_node->get_variable();
    variable->update_data_type(quantization_attrs.quantization_dt);

    auto variable_infos = get_variable_infos(past_rv_node->get_variable()->get_info(), quantization_attrs);
    auto new_past_rv_node = std::make_shared<ov::intel_gpu::op::ReadValues>();

    if (past_rv_node->get_input_size() == 0) {
        new_past_rv_node = std::make_shared<ov::intel_gpu::op::ReadValues>(past_rv_node->get_variable(), variable_infos);
    } else {
        auto initializer_dq = std::make_shared<ov::op::internal::DynamicQuantize>(past_rv_node->input_value(0),
                                                                                  quantization_attrs);
        initializer_dq->set_friendly_name(past_rv_node->input_value(0).get_node_shared_ptr()->get_friendly_name() + "_dyn_quan");
        ov::copy_runtime_info(past_rv_node->input_value(0).get_node_shared_ptr(), initializer_dq);

        OutputVector initializer_outputs = { initializer_dq->output(0), initializer_dq->output(1) };

        if (quantization_attrs.quantization_type == ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric &&
            quantization_attrs.output_storage_type == ov::op::internal::DynamicQuantize::OutputStorageType::Planar) {
            initializer_outputs.push_back(initializer_dq->output(2));
        }

        new_past_rv_node = std::make_shared<ov::intel_gpu::op::ReadValues>(initializer_outputs, past_rv_node->get_variable(), variable_infos);
    }

    ov::copy_runtime_info(past_rv_node, new_past_rv_node);
    past_rv_node->output(0).replace(new_past_rv_node->output(0));

    return new_past_rv_node;
}

std::shared_ptr<ov::intel_gpu::op::KVCacheCompressed>
    update_kv_cache(std::shared_ptr<ov::intel_gpu::op::ReadValue> past_rv_node,
                    std::shared_ptr<ov::intel_gpu::op::KVCache> kv_cache_node,
                    const ov::op::internal::DynamicQuantize::Attributes& quantization_attrs) {
    OutputVector kv_cache_inputs = { past_rv_node->output(0),
                                     kv_cache_node->input_value(1),
                                     kv_cache_node->input_value(2) };
    if (kv_cache_node->get_trim()) {
        kv_cache_inputs.push_back(kv_cache_node->input_value(3));
    }
    kv_cache_inputs.push_back(past_rv_node->output(1));

    if (quantization_attrs.quantization_type == ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric &&
        quantization_attrs.output_storage_type == ov::op::internal::DynamicQuantize::OutputStorageType::Planar) {
        kv_cache_inputs.push_back(past_rv_node->output(2));
    }

    auto new_kv_cache = std::make_shared<op::KVCacheCompressed>(kv_cache_inputs,
                                                                kv_cache_node->get_variable(),
                                                                kv_cache_node->get_trim(),
                                                                kv_cache_node->get_concat_axis(),
                                                                kv_cache_node->get_gather_axis(),
                                                                quantization_attrs);

    new_kv_cache->set_friendly_name(kv_cache_node->get_friendly_name());
    ov::copy_runtime_info(kv_cache_node, new_kv_cache);

    return new_kv_cache;
}
}  // namespace

class KVCacheCompressionMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("KVCacheCompressionMatcher");
    KVCacheCompressionMatcher(ov::element::Type compression_dt, bool supports_immad);
};

KVCacheCompressionMatcher::KVCacheCompressionMatcher(ov::element::Type compression_dt, bool supports_immad) {
    using namespace ov::pass::pattern;

    if (!cldnn::one_of(compression_dt, {element::i8, element::u8, element::i4, element::u4})) {
        return;
    }

    const auto quantization_type = ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;
    const auto output_storage_type = supports_immad ? ov::op::internal::DynamicQuantize::OutputStorageType::Planar
                                                    : ov::op::internal::DynamicQuantize::OutputStorageType::InterleavedScalesZP;
    bool combine_scales_and_zp = output_storage_type == ov::op::internal::DynamicQuantize::OutputStorageType::InterleavedScalesZP;
    GPU_DEBUG_LOG << "KV-cache compression configuration: "
                  << "dt=" << compression_dt << ", "
                  << "asym=" << (quantization_type == ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric) << ", "
                  << "single_buffer_for_scales_and_zp=" << combine_scales_and_zp << "\n";

    auto query = any_input();
    auto past_seq_len = any_input();

    auto key_past = wrap_type<ov::intel_gpu::op::ReadValue>();
    auto key_new_token = any_input();
    auto key_beam_idx = any_input();
    auto key_cache_notrim = wrap_type<ov::intel_gpu::op::KVCache>({key_past, key_new_token, key_beam_idx});
    auto key_cache_trim = wrap_type<ov::intel_gpu::op::KVCache>({key_past, key_new_token, key_beam_idx, past_seq_len});
    auto key_cache = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{key_cache_notrim, key_cache_trim});

    auto value_past = wrap_type<ov::intel_gpu::op::ReadValue>();
    auto value_new_token = any_input();
    auto value_beam_idx = any_input();
    auto value_cache_notrim = wrap_type<ov::intel_gpu::op::KVCache>({value_past, value_new_token, value_beam_idx});
    auto value_cache_trim = wrap_type<ov::intel_gpu::op::KVCache>({value_past, value_new_token, value_beam_idx, past_seq_len});
    auto value_cache = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{value_cache_notrim, value_cache_trim});

    auto input_attn_mask = any_input();
    auto input_scale = any_input();
    auto input_beam_table = any_input();

    auto sdpa_without_attn_mask_m = wrap_type<ov::intel_gpu::op::IndirectSDPA>({ query, key_cache, value_cache, input_beam_table });
    auto sdpa_with_attn_mask_m = wrap_type<ov::intel_gpu::op::IndirectSDPA>({ query, key_cache, value_cache, input_attn_mask, input_beam_table });
    auto sdpa_with_attn_mask_and_scale_m =
        wrap_type<ov::intel_gpu::op::IndirectSDPA>({ query, key_cache, value_cache, input_attn_mask, input_scale, input_beam_table });

    auto sdpa = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{sdpa_without_attn_mask_m, sdpa_with_attn_mask_m, sdpa_with_attn_mask_and_scale_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        const auto& pattern_map = m.get_pattern_value_map();

        auto query_node = pattern_map.at(query).get_node_shared_ptr();

        auto key_new_token_node = pattern_map.at(key_new_token).get_node_shared_ptr();
        auto key_cache_node = ov::as_type_ptr<ov::intel_gpu::op::KVCache>(pattern_map.at(key_cache).get_node_shared_ptr());
        auto value_cache_node = ov::as_type_ptr<ov::intel_gpu::op::KVCache>(pattern_map.at(value_cache).get_node_shared_ptr());
        if (!key_cache_node->get_indirect() || !value_cache_node->get_indirect()) {
            return false;
        }
        auto sdpa_node = ov::as_type_ptr<ov::intel_gpu::op::IndirectSDPA>(m.get_match_root());

        auto key_past_rv_node = ov::as_type_ptr<ov::intel_gpu::op::ReadValue>(pattern_map.at(key_past).get_node_shared_ptr());
        auto value_past_rv_node = ov::as_type_ptr<ov::intel_gpu::op::ReadValue>(pattern_map.at(value_past).get_node_shared_ptr());

        auto data_rank = key_cache_node->get_input_partial_shape(0).size();
        auto get_shape_group_sizes = [&](const std::vector<int64_t>& transposed_order) {
            std::vector<uint64_t> group_sizes(data_rank, 1);
            std::vector<int64_t> order = transposed_order;
            if (transposed_order.size() != data_rank) {
                order.resize(data_rank);
                std::iota(order.begin(), order.end(), 0);
            }

            group_sizes[order[data_rank - 1]] = UINT64_MAX;

            return group_sizes;
        };

        // Reorder scales in static order: [batch, num_heads, seq_len, head_size]
        auto get_scales_output_order = [&](const std::vector<int64_t>& transposed_order) {
            std::vector<uint64_t> scales_zp_output_order(data_rank);
            scales_zp_output_order[0] = transposed_order[0];
            scales_zp_output_order[1] = transposed_order[1];
            scales_zp_output_order[2] = transposed_order[2];
            scales_zp_output_order[3] = transposed_order[3];

            return scales_zp_output_order;
        };

        ov::op::internal::DynamicQuantize::Attributes config;
        config.quantization_type = quantization_type;
        config.group_sizes = get_shape_group_sizes(sdpa_node->get_input1_transpose_order());
        config.quantization_dt = element::i8;
        const auto scale_dt = query_node->get_output_element_type(0);
        config.scale_dt = scale_dt;
        config.scales_zp_output_order = get_scales_output_order(sdpa_node->get_input1_transpose_order());
        config.output_storage_type = output_storage_type;

        if (config.quantization_type == ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric) {
            // For INT4 KV cache, zero-point must be stored as fp16 (not integer).
            // u4 has only 16 quantization levels, so rounding ZP to integer
            // introduces a large systematic bias: error = (zp - round(zp)) * scale
            // applied uniformly to every element in the group.
            bool is_int4 = ov::element::Type(compression_dt).bitwidth() == 4;
            config.zp_dt = (supports_immad && !is_int4) ? element::i8 : scale_dt;
        }

        key_past_rv_node = update_past_read_value(key_past_rv_node, config);
        value_past_rv_node = update_past_read_value(value_past_rv_node, config);

        auto new_key_cache = update_kv_cache(key_past_rv_node, key_cache_node, config);
        auto new_value_cache = update_kv_cache(value_past_rv_node, value_cache_node, config);

        OutputVector sdpa_inputs;
        // Add Query, Key, Value, attention_mask, scale inputs
        for (size_t i = 0; i < sdpa_node->get_input_size() - 1; i++) {
            sdpa_inputs.push_back(sdpa_node->input_value(i));
        }

        // Replace Key and Value inputs with compressed ones
        sdpa_inputs[1] = new_key_cache->output(0);
        sdpa_inputs[2] = new_value_cache->output(0);

        // Add Key and Value compression scales
        sdpa_inputs.push_back(new_key_cache->output(2));
        sdpa_inputs.push_back(new_value_cache->output(2));

        // Add Key and Value compression zero points
        if (config.quantization_type == ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric &&
            config.output_storage_type == ov::op::internal::DynamicQuantize::OutputStorageType::Planar) {
            sdpa_inputs.push_back(new_key_cache->output(3));
            sdpa_inputs.push_back(new_value_cache->output(3));
        }

        auto input0_transpose_order = sdpa_node->get_input0_transpose_order();
        auto input1_transpose_order = sdpa_node->get_input1_transpose_order();
        auto input2_transpose_order = sdpa_node->get_input2_transpose_order();
        auto output_transpose_order = sdpa_node->get_output_transpose_order();

        auto new_sdpa = std::make_shared<op::IndirectSDPA>(sdpa_inputs,
                                                           new_key_cache->output(1),
                                                           sdpa_node->get_causal(),
                                                           sdpa_node->get_indirect_axis(),
                                                           input0_transpose_order,
                                                           input1_transpose_order,
                                                           input2_transpose_order,
                                                           output_transpose_order,
                                                           config,
                                                           sdpa_node->get_output_type());

        new_key_cache->set_friendly_name(key_cache_node->get_friendly_name());
        ov::copy_runtime_info(key_cache_node, new_key_cache);

        new_value_cache->set_friendly_name(value_cache_node->get_friendly_name());
        ov::copy_runtime_info(value_cache_node, new_value_cache);

        new_sdpa->set_friendly_name(sdpa_node->get_friendly_name());
        ov::copy_runtime_info(sdpa_node, new_sdpa);

        ov::replace_node(sdpa_node, new_sdpa);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sdpa, "KVCacheCompressionMatcher");
    this->register_matcher(m, callback);
}

/// Compresses the key / value caches of the GPU SparseSDPA op made indirect by IndirectSparseAttentionOpt and replaces
/// the op with a SparseSDPA that reads the compressed caches. The caches are quantized like the SDPA ones on systolic
/// platforms (planar storage): per token, asymmetric, with f16 scales and i8 zero points (f16 for a 4-bit cache), as
/// SparseSDPA reads them (see sparse_kv_quant.hpp). SparseSDPA does not read the interleaved scales / zero points of the
/// other platforms, so there the caches stay uncompressed (with a warning).
class KVCacheCompressionSparseMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("KVCacheCompressionSparseMatcher");
    KVCacheCompressionSparseMatcher(ov::element::Type compression_dt, bool supports_immad);
};

KVCacheCompressionSparseMatcher::KVCacheCompressionSparseMatcher(ov::element::Type compression_dt, bool supports_immad) {
    using namespace ov::pass::pattern;
    using SparseSDPA = ov::intel_gpu::op::SparseSDPA;
    using In = SparseSDPA::InputIdx;

    if (!cldnn::one_of(compression_dt, {element::i8, element::u8, element::i4, element::u4})) {
        return;
    }

    auto sparse_sdpa = wrap_type<SparseSDPA>();
    auto warned = std::make_shared<bool>(false);

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto attention = ov::as_type_ptr<SparseSDPA>(m.get_match_root());
        // A compressed SparseSDPA has no sinks input.
        if (!attention || transformation_callback(attention) || attention->get_config().kv_compressed() || attention->has_sinks() ||
            attention->get_input_element_type(In::QUERY) != element::f16) {
            return false;
        }
        std::shared_ptr<op::KVCache> kv_caches[2];
        std::shared_ptr<op::ReadValue> past_read_values[2];
        for (size_t i = 0; i < 2; ++i) {
            kv_caches[i] = ov::as_type_ptr<op::KVCache>(attention->get_input_node_shared_ptr(i == 0 ? In::KEY : In::VALUE));
            // Shape reads (the key_length of a SparseAttention without one) move to the compressed cache below.
            if (!kv_caches[i] || ov::is_type<op::KVCacheCompressed>(kv_caches[i]) || !kv_caches[i]->get_indirect() ||
                kv_caches[i]->get_rt_info().count(op::KV_CACHE_REQUIRES_IDENTITY_BEAM_IDX) == 0 || !read_only_by(kv_caches[i]->output(0), attention.get())) {
                return false;
            }
            past_read_values[i] = ov::as_type_ptr<op::ReadValue>(kv_caches[i]->get_input_node_shared_ptr(0));
            if (!past_read_values[i]) {
                return false;
            }
        }
        if (!supports_immad) {
            if (!*warned) {
                *warned = true;
                OPENVINO_WARN("[GPU] KV cache compression is not supported for SparseAttention on this device: its KV caches stay uncompressed.");
            }
            return false;
        }

        const bool four_bit = compression_dt.bitwidth() == 4;
        if (four_bit) {
            // The kv_cache packs 4-bit codes two per byte, which halves the inner dimension of its layout. A standalone
            // DynamicQuantize of a ReadValue initializer does not pack, so the initializer must be empty, and a shape read
            // of the compressed cache would see the packed width.
            for (size_t i = 0; i < 2; ++i) {
                const auto& past = past_read_values[i];
                bool empty_initializer = past->get_input_size() == 0;
                if (!empty_initializer) {
                    const auto& shape = past->get_input_partial_shape(0);
                    if (shape.rank().is_static()) {
                        const auto axis = static_cast<size_t>(ov::util::normalize(kv_caches[i]->get_concat_axis(), shape.rank().get_length()));
                        empty_initializer = shape[axis].is_static() && shape[axis].get_length() == 0;
                    }
                }
                if (!empty_initializer || kv_caches[i]->get_output_target_inputs(0).size() != 1) {
                    GPU_DEBUG_LOG << "SparseAttention " << attention->get_friendly_name()
                                  << " keeps uncompressed KV caches: a 4-bit cache needs an empty initializer and no shape reads" << std::endl;
                    return false;
                }
            }
        }

        // The quantized histories have the head sizes of the uncompressed ones, which SparseSDPA already validated.
        op::KVQuant quant;
        quant.mode = op::KVQuant::Mode::BY_TOKEN;
        quant.bits = four_bit ? 4 : 8;
        auto sdpa_config = attention->get_config();
        sdpa_config.key_quant = quant;
        sdpa_config.value_quant = quant;

        ov::op::internal::DynamicQuantize::Attributes config;
        config.quantization_type = ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;
        config.group_sizes = {1, 1, 1, UINT64_MAX};  // per token of each head: [B, Hkv, S, 1] parameters
        config.quantization_dt = element::i8;
        config.scale_dt = element::f16;
        config.zp_dt = quant.bits == 4 ? element::f16 : element::i8;
        config.scales_zp_output_order = {0, 1, 2, 3};
        config.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::Planar;

        std::shared_ptr<op::KVCacheCompressed> compressed[2];
        for (size_t i = 0; i < 2; ++i) {
            auto read_values = update_past_read_value(past_read_values[i], config);
            compressed[i] = update_kv_cache(read_values, kv_caches[i], config);  // keeps the runtime info of the cache
            // The 8-bit compressed data has the shape of the cache (a 4-bit one has no shape reads, see above).
            for (const auto& input : kv_caches[i]->output(0).get_target_inputs()) {
                if (input.get_node() != attention.get()) {
                    input.replace_source_output(compressed[i]->output(0));
                }
            }
        }

        ov::OutputVector inputs{attention->input_value(In::QUERY),
                                compressed[0]->output(0),
                                compressed[1]->output(0),
                                attention->input_value(In::SELECTED_BLOCK_INDICES),
                                attention->input_value(In::SELECTED_BLOCK_COUNT),
                                attention->input_value(In::KEY_LENGTH),
                                compressed[0]->output(2),
                                compressed[1]->output(2),
                                compressed[0]->output(3),
                                compressed[1]->output(3)};
        if (attention->has_scale_input()) {
            inputs.push_back(attention->input_value(attention->get_config().scale_input_idx()));
        }
        const auto compressed_sdpa = std::make_shared<SparseSDPA>(inputs, sdpa_config, attention->get_output_type());
        compressed_sdpa->set_friendly_name(attention->get_friendly_name());
        ov::copy_runtime_info(attention, compressed_sdpa);
        // As for SDPA, the uncompressed KVCache nodes are no longer read and drop out of the model.
        ov::replace_node(attention, compressed_sdpa);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sparse_sdpa, "KVCacheCompressionSparseMatcher");
    this->register_matcher(m, callback);
}

bool KVCacheCompression::run_on_model(const std::shared_ptr<ov::Model>& m) {
    return pass::GraphRewrite::run_on_model(m);
}

KVCacheCompression::KVCacheCompression(ov::element::Type compression_dt, bool supports_immad) {
    add_matcher<ov::intel_gpu::KVCacheCompressionMatcher>(compression_dt, supports_immad);
    add_matcher<ov::intel_gpu::KVCacheCompressionSparseMatcher>(compression_dt, supports_immad);
}

}  // namespace ov::intel_gpu
