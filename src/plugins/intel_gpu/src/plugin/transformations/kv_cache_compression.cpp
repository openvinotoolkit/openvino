// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kv_cache_compression.hpp"

#include <memory>
#include <optional>

#include "intel_gpu/op/indirect_sdpa.hpp"
#include "intel_gpu/op/kv_cache.hpp"
#include "intel_gpu/op/kv_cache_compressed.hpp"
#include "intel_gpu/op/read_value.hpp"
#include "intel_gpu/op/read_values.hpp"
#include "intel_gpu/op/sparse_sdpa.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/sink.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "ov_ops/dynamic_quantize.hpp"
#include "transformations/utils/utils.hpp"

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

// Compressed counterpart of a plain (non-indirect) KVCache, for consumers that do not read a beam table.
std::shared_ptr<ov::intel_gpu::op::KVCacheCompressed> make_compressed_kv_cache(std::shared_ptr<ov::intel_gpu::op::ReadValue> past_rv_node,
                                                                               std::shared_ptr<ov::intel_gpu::op::KVCache> kv_cache_node,
                                                                               const ov::Output<ov::Node>& beam_idx,
                                                                               int64_t gather_axis,
                                                                               const ov::op::internal::DynamicQuantize::Attributes& quantization_attrs) {
    OutputVector inputs = {past_rv_node->output(0), kv_cache_node->input_value(1), beam_idx, past_rv_node->output(1), past_rv_node->output(2)};
    auto new_kv_cache = std::make_shared<op::KVCacheCompressed>(inputs,
                                                                kv_cache_node->get_variable(),
                                                                false,
                                                                kv_cache_node->get_concat_axis(),
                                                                gather_axis,
                                                                quantization_attrs);
    new_kv_cache->set_friendly_name(kv_cache_node->get_friendly_name());
    ov::copy_runtime_info(kv_cache_node, new_kv_cache);
    return new_kv_cache;
}

// A KVCache feeding SparseSDPA, reduced to what the compression needs.
struct SparseSDPACache {
    std::shared_ptr<ov::intel_gpu::op::KVCache> kv_cache;
    std::shared_ptr<ov::intel_gpu::op::ReadValue> past;
    ov::Output<ov::Node> beam_idx;
    int64_t gather_axis = 0;
};

// Accepts `KVCache(ReadValue, new)` of a single sequence and `KVCache(Gather(ReadValue, beam_idx), new)`.
std::optional<SparseSDPACache> analyze_sparse_sdpa_cache(const ov::Output<ov::Node>& cache_output) {
    auto kv_cache = ov::as_type_ptr<ov::intel_gpu::op::KVCache>(cache_output.get_node_shared_ptr());
    if (!kv_cache || cache_output.get_index() != 0 || ov::is_type<ov::intel_gpu::op::KVCacheCompressed>(kv_cache) || kv_cache->get_indirect() ||
        kv_cache->get_trim() || kv_cache->get_update_kv() || kv_cache->get_input_size() != 2) {
        return std::nullopt;
    }
    constexpr size_t rank = 4;  // [B, Hkv, S, D]
    const auto& past_ps = kv_cache->get_input_partial_shape(0);
    if (past_ps.rank().is_dynamic() || past_ps.size() != rank || ov::util::normalize(kv_cache->get_concat_axis(), static_cast<int64_t>(rank)) != 2 ||
        kv_cache->get_output_element_type(0) != ov::element::f16) {
        return std::nullopt;
    }

    SparseSDPACache res;
    res.kv_cache = kv_cache;
    const auto past_node = kv_cache->input_value(0).get_node_shared_ptr();
    if (auto gather = ov::as_type_ptr<ov::op::v8::Gather>(past_node)) {
        res.past = ov::as_type_ptr<ov::intel_gpu::op::ReadValue>(gather->input_value(0).get_node_shared_ptr());
        if (!res.past || !ov::is_type<ov::op::v0::Parameter>(gather->input_value(1).get_node()) || gather->get_batch_dims() != 0 || gather->get_axis() != 0 ||
            gather->output(0).get_target_inputs().size() != 1) {
            return std::nullopt;
        }
        res.beam_idx = gather->input_value(1);
    } else {
        // Without a gather the cache belongs to one sequence, which a constant beam index expresses.
        res.past = ov::as_type_ptr<ov::intel_gpu::op::ReadValue>(past_node);
        if (!res.past || !past_ps[0].is_static() || past_ps[0].get_length() != 1) {
            return std::nullopt;
        }
        res.beam_idx = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
    }
    if (res.past->get_output_element_type(0) != ov::element::f16 || res.past->output(0).get_target_inputs().size() != 1) {
        return std::nullopt;
    }
    return res;
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

bool KVCacheCompression::run_on_model(const std::shared_ptr<ov::Model>& m) {
    return pass::GraphRewrite::run_on_model(m);
}

// SparseSDPA reads K/V as dense [B, Hkv, S, D] states without a beam table, so its caches are compressed as per-token
// asymmetric i8 with planar scales / zero points; the beam table output of the compressed KVCache stays unused.
class SparseSDPAKVCacheCompressionMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SparseSDPAKVCacheCompressionMatcher");
    SparseSDPAKVCacheCompressionMatcher(ov::element::Type compression_dt, bool supports_immad);
};

SparseSDPAKVCacheCompressionMatcher::SparseSDPAKVCacheCompressionMatcher(ov::element::Type compression_dt, bool supports_immad) {
    using namespace ov::pass::pattern;
    using In = ov::intel_gpu::op::SparseSDPA::InputIdx;

    // The kernels read i8 data with planar f16 scales and i8 zero points (immad only); 4-bit caches stay f16.
    if (!supports_immad || !cldnn::one_of(compression_dt, {element::i8, element::u8})) {
        return;
    }

    auto sdpa_m = wrap_type<ov::intel_gpu::op::SparseSDPA>();

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }
        auto sdpa = ov::as_type_ptr<ov::intel_gpu::op::SparseSDPA>(m.get_match_root());
        if (!sdpa) {
            return false;
        }
        const auto sdpa_config = sdpa->get_config();
        if (sdpa_config.kv_compressed || sdpa->has_sinks() || sdpa->get_input_element_type(In::QUERY) != element::f16) {
            return false;
        }

        auto key = analyze_sparse_sdpa_cache(sdpa->input_value(In::KEY));
        auto value = analyze_sparse_sdpa_cache(sdpa->input_value(In::VALUE));
        if (!key || !value || key->kv_cache == value->kv_cache || key->past == value->past) {
            return false;
        }

        ov::op::internal::DynamicQuantize::Attributes config;
        config.quantization_type = ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;
        config.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::Planar;
        config.group_sizes = {1, 1, 1, UINT64_MAX};  // one scale / zero point per token row
        config.quantization_dt = element::i8;
        config.scale_dt = element::f16;
        config.zp_dt = element::i8;
        config.scales_zp_output_order = {0, 1, 2, 3};

        auto key_rv = update_past_read_value(key->past, config);
        auto value_rv = update_past_read_value(value->past, config);
        auto new_key_cache = make_compressed_kv_cache(key_rv, key->kv_cache, key->beam_idx, key->gather_axis, config);
        auto new_value_cache = make_compressed_kv_cache(value_rv, value->kv_cache, value->beam_idx, value->gather_axis, config);

        // The kernel needs a constant scale input; fold the default one in when the op relied on it.
        ov::Output<ov::Node> scale;
        if (sdpa->has_scale_input()) {
            scale = sdpa->input_value(In::SCALE);
        } else {
            const float default_scale = 1.0f / std::sqrt(static_cast<float>(sdpa_config.k_head_size));
            scale = ov::op::v0::Constant::create(element::f32, ov::Shape{}, {default_scale});
        }

        ov::OutputVector args(In::COUNT_COMPRESSED);
        for (size_t i : {In::QUERY, In::SELECTED_BLOCK_INDICES, In::SELECTED_BLOCK_COUNT, In::KEY_LENGTH}) {
            args[i] = sdpa->input_value(i);
        }
        args[In::KEY] = new_key_cache->output(0);
        args[In::VALUE] = new_value_cache->output(0);
        args[In::SCALE] = scale;
        args[In::KEY_SCALE] = new_key_cache->output(2);
        args[In::VALUE_SCALE] = new_value_cache->output(2);
        args[In::KEY_ZP] = new_key_cache->output(3);
        args[In::VALUE_ZP] = new_value_cache->output(3);

        auto compressed_config = sdpa_config;
        compressed_config.kv_compressed = true;
        auto new_sdpa = std::make_shared<ov::intel_gpu::op::SparseSDPA>(args, compressed_config, sdpa->get_output_type());
        new_sdpa->set_friendly_name(sdpa->get_friendly_name());
        ov::copy_runtime_info(sdpa, new_sdpa);
        ov::replace_node(sdpa, new_sdpa);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sdpa_m, "SparseSDPAKVCacheCompressionMatcher");
    this->register_matcher(m, callback);
}

KVCacheCompression::KVCacheCompression(ov::element::Type compression_dt, bool supports_immad) {
    add_matcher<ov::intel_gpu::KVCacheCompressionMatcher>(compression_dt, supports_immad);
    add_matcher<ov::intel_gpu::SparseSDPAKVCacheCompressionMatcher>(compression_dt, supports_immad);
}

}  // namespace ov::intel_gpu
