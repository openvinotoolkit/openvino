// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kv_cache_compression.hpp"

#include "intel_gpu/op/kv_cache.hpp"
#include "intel_gpu/op/indirect_sdpa.hpp"
#include "intel_gpu/op/read_value.hpp"
#include "intel_gpu/op/read_values.hpp"
#include "intel_gpu/op/dynamic_quantize.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#include "openvino/core/node_vector.hpp"
#include "openvino/core/rt_info.hpp"
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
#include "transformations/utils/utils.hpp"

#include <memory>

namespace ov {
namespace intel_gpu {

namespace {
std::vector<ov::op::util::VariableInfo> get_variable_infos(const ov::op::util::VariableInfo& data_variable_info,
                                                           const ov::op::internal::QuantizationConfig& config,
                                                           const std::vector<uint64_t>& scales_zp_output_order,
                                                           const bool combine_scales_and_zp = false) {
    std::vector<ov::op::util::VariableInfo> infos;

    // Add initial data variable info
    infos.push_back(data_variable_info);

    // Infer DQ shapes
    ov::intel_gpu::op::DynamicQuantize dq;
    auto dq_shapes =
        ov::intel_gpu::op::DynamicQuantize::shape_infer(&dq, {data_variable_info.data_shape}, config, scales_zp_output_order, combine_scales_and_zp);

    const auto variable_id = data_variable_info.variable_id;
    const auto scale_shape = dq_shapes[1];
    const auto scale_dt = config.scale_dt;

    // Add scales variable info
    infos.push_back(ov::op::util::VariableInfo{scale_shape, scale_dt, variable_id});

    if (config.is_asymmetric_quantization() && !combine_scales_and_zp) {
        // Add zero points variable info
        const auto zp_dt = config.zp_dt;
        infos.push_back(ov::op::util::VariableInfo{scale_shape, zp_dt, variable_id});
    }

    return infos;
}

std::shared_ptr<ov::intel_gpu::op::ReadValues>
    update_past_read_value(std::shared_ptr<ov::intel_gpu::op::ReadValue> past_rv_node,
                           const ov::op::internal::QuantizationConfig& config,
                           const std::vector<uint64_t>& scales_zp_output_order,
                           const bool combine_scales_and_zp = false) {
    auto variable = past_rv_node->get_variable();
    variable->update_data_type(config.quantization_dt);

    auto variable_infos = get_variable_infos(past_rv_node->get_variable()->get_info(), config, scales_zp_output_order, combine_scales_and_zp);
    auto new_past_rv_node = std::make_shared<ov::intel_gpu::op::ReadValues>();

    if (past_rv_node->get_input_size() == 0) {
        new_past_rv_node = std::make_shared<ov::intel_gpu::op::ReadValues>(past_rv_node->get_variable(), variable_infos);
    } else {
        auto initializer_dq = std::make_shared<ov::intel_gpu::op::DynamicQuantize>(past_rv_node->get_input_node_shared_ptr(0),
                                                                                   config,
                                                                                   scales_zp_output_order,
                                                                                   combine_scales_and_zp);
        initializer_dq->set_friendly_name(past_rv_node->get_input_node_shared_ptr(0)->get_friendly_name() + "_dyn_quan");
        ov::copy_runtime_info(past_rv_node->get_input_node_shared_ptr(0), initializer_dq);

        OutputVector initializer_outputs = { initializer_dq->output(0), initializer_dq->output(1) };

        if (config.is_asymmetric_quantization() && !combine_scales_and_zp)
            initializer_outputs.push_back(initializer_dq->output(2));

        new_past_rv_node = std::make_shared<ov::intel_gpu::op::ReadValues>(initializer_outputs, past_rv_node->get_variable(), variable_infos);
    }

    ov::copy_runtime_info(past_rv_node, new_past_rv_node);
    past_rv_node->output(0).replace(new_past_rv_node->output(0));

    return new_past_rv_node;
}

std::shared_ptr<ov::intel_gpu::op::KVCache>
    update_kv_cache(std::shared_ptr<ov::intel_gpu::op::ReadValue> past_rv_node,
                    std::shared_ptr<ov::intel_gpu::op::KVCache> kv_cache_node,
                    const ov::op::internal::QuantizationConfig& config,
                    const std::vector<uint64_t>& scales_zp_output_order,
                    const bool combine_scales_and_zp = false) {
    OutputVector kv_cache_inputs = { past_rv_node->output(0),
                                     kv_cache_node->get_input_node_shared_ptr(1),
                                     kv_cache_node->get_input_node_shared_ptr(2),
                                     past_rv_node->output(1) };

    if (config.is_asymmetric_quantization() && !combine_scales_and_zp)
        kv_cache_inputs.push_back(past_rv_node->output(2));

    auto new_kv_cache = std::make_shared<op::KVCache>(kv_cache_inputs,
                                                      kv_cache_node->get_variable(),
                                                      kv_cache_node->get_concat_axis(),
                                                      kv_cache_node->get_gather_axis(),
                                                      combine_scales_and_zp,
                                                      config,
                                                      scales_zp_output_order);

    new_kv_cache->set_friendly_name(kv_cache_node->get_friendly_name());
    ov::copy_runtime_info(kv_cache_node, new_kv_cache);

    return new_kv_cache;
}
}  // namespace

class KVCacheCompressionMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("KVCacheCompressionMatcher", "0");
    KVCacheCompressionMatcher(ov::element::Type compression_dt);
};

KVCacheCompressionMatcher::KVCacheCompressionMatcher(ov::element::Type compression_dt) {
    using namespace ov::pass::pattern;

    if (compression_dt != element::i8)
        return;

    auto quantization_type = ov::op::internal::QuantizationConfig::QuantizationType::Asymmetric;
    bool combine_scales_and_zp = quantization_type == ov::op::internal::QuantizationConfig::QuantizationType::Asymmetric;

    GPU_DEBUG_LOG << "KV-cache compression configuration: "
                  << "dt=" << compression_dt << ", "
                  << "asym=" << (quantization_type == ov::op::internal::QuantizationConfig::QuantizationType::Asymmetric) << ", "
                  << "single_buffer_for_scales_and_zp=" << combine_scales_and_zp << "\n";

    auto query = any_input();

    auto key_past = wrap_type<ov::intel_gpu::op::ReadValue>();
    auto key_new_token = any_input();
    auto key_beam_idx = any_input();
    auto key_cache = wrap_type<ov::intel_gpu::op::KVCache>({key_past, key_new_token, key_beam_idx});

    auto value_past = wrap_type<ov::intel_gpu::op::ReadValue>();
    auto value_new_token = any_input();
    auto value_beam_idx = any_input();
    auto value_cache = wrap_type<ov::intel_gpu::op::KVCache>({value_past, value_new_token, value_beam_idx});

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
        auto key_cache_node = std::dynamic_pointer_cast<ov::intel_gpu::op::KVCache>(pattern_map.at(key_cache).get_node_shared_ptr());
        auto value_cache_node = std::dynamic_pointer_cast<ov::intel_gpu::op::KVCache>(pattern_map.at(value_cache).get_node_shared_ptr());
        auto sdpa_node = std::dynamic_pointer_cast<ov::intel_gpu::op::IndirectSDPA>(m.get_match_root());

        auto key_past_rv_node = std::dynamic_pointer_cast<ov::intel_gpu::op::ReadValue>(pattern_map.at(key_past).get_node_shared_ptr());
        auto value_past_rv_node = std::dynamic_pointer_cast<ov::intel_gpu::op::ReadValue>(pattern_map.at(value_past).get_node_shared_ptr());

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

        auto group_sizes = get_shape_group_sizes(sdpa_node->get_input1_transpose_order());
        auto scales_zp_output_order = get_scales_output_order(sdpa_node->get_input1_transpose_order());

        ov::op::internal::QuantizationConfig config;
        config.type = quantization_type;
        config.group_sizes = group_sizes;
        config.quantization_dt = element::i8;
        config.scale_dt = query_node->get_output_element_type(0);

        if (config.is_asymmetric_quantization())
            config.zp_dt = query_node->get_output_element_type(0);

        key_past_rv_node = update_past_read_value(key_past_rv_node, config, scales_zp_output_order, combine_scales_and_zp);
        value_past_rv_node = update_past_read_value(value_past_rv_node, config, scales_zp_output_order, combine_scales_and_zp);

        auto new_key_cache = update_kv_cache(key_past_rv_node, key_cache_node, config, scales_zp_output_order, combine_scales_and_zp);
        auto new_value_cache = update_kv_cache(value_past_rv_node, value_cache_node, config, scales_zp_output_order, combine_scales_and_zp);

        OutputVector sdpa_inputs;
        // Add Query, Key, Value, attention_mask, scale inputs
        for (size_t i = 0; i < sdpa_node->get_input_size() - 1; i++)
            sdpa_inputs.push_back(sdpa_node->get_input_node_shared_ptr(i));

        // Replace Key and Value inputs with compressed ones
        sdpa_inputs[1] = new_key_cache->output(0);
        sdpa_inputs[2] = new_value_cache->output(0);

        // Add Key and Value compression scales
        sdpa_inputs.push_back(new_key_cache->output(2));
        sdpa_inputs.push_back(new_value_cache->output(2));

        // Add Key and Value compression zero points
        if (config.is_asymmetric_quantization() && !combine_scales_and_zp) {
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
                                                           combine_scales_and_zp,
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

KVCacheCompression::KVCacheCompression(ov::element::Type compression_dt) {
    add_matcher<ov::intel_gpu::KVCacheCompressionMatcher>(compression_dt);
}

}  // namespace intel_gpu
}  // namespace ov
