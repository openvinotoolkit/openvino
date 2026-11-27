// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kv_cache_compression.hpp"

#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/kv_cache.hpp"
#include "intel_gpu/op/kv_cache_compressed.hpp"
#include "intel_gpu/op/indirect_sdpa.hpp"
#include "intel_gpu/op/read_value.hpp"
#include "intel_gpu/op/read_values.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "ov_ops/dynamic_quantize.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"

#include "openvino/core/node_vector.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/op/sink.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "transformations/utils/utils.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_set>
#include "openvino/core/graph_util.hpp"


namespace ov::intel_gpu {

namespace {
struct KBiasAnalysis {
    float max_abs = 0.f;
    bool found = false;
    bool complete = true;
    bool budget_exhausted = false;
};

// Inspect projection biases only along supported K data edges, never shape/position inputs.
// This is a static heuristic: a large projection bias is a known trigger for K outlier channels that int8 per-token
// quantization resolved poorly (CVS-190235). It does not model the K value range, so it can both over- and
// under-report (uniform offsets, weight/input-driven outliers, post-bias scaling/normalization/RoPE).
KBiasAnalysis analyze_k_bias(const std::shared_ptr<ov::Node>& new_token) {
    KBiasAnalysis result;
    auto skip_convert = [](std::shared_ptr<ov::Node> n) {
        while (ov::is_type<ov::op::v0::Convert>(n)) {
            n = n->get_input_node_shared_ptr(0);
        }
        return n;
    };
    auto record_bias = [&](const std::shared_ptr<ov::Node>& n) {
        // Restrict folding to a Constant optionally wrapped in Converts.
        if (!ov::is_type<ov::op::v0::Constant>(skip_convert(n))) {
            result.complete = false;
            return;
        }
        auto c = ov::util::get_constant_from_source(n->output(0));
        if (!c) {
            result.complete = false;
            return;
        }
        result.found = true;
        for (auto v : c->cast_vector<float>()) {
            result.max_abs = std::max(result.max_abs, std::abs(v));
        }
    };
    auto is_linear = [](const std::shared_ptr<ov::Node>& n) {
        return ov::is_type<ov::op::v0::MatMul>(n) || ov::is_type<ov::intel_gpu::op::FullyConnected>(n);
    };

    std::vector<std::shared_ptr<ov::Node>> stack{new_token};
    std::unordered_set<const ov::Node*> visited;
    constexpr size_t node_budget = 64;
    while (!stack.empty()) {
        auto node = stack.back();
        stack.pop_back();
        if (!visited.insert(node.get()).second) {
            continue;
        }
        if (visited.size() > node_budget) {
            result.complete = false;
            result.budget_exhausted = true;
            break;
        }
        if (is_linear(node)) {
            if (ov::is_type<ov::intel_gpu::op::FullyConnected>(node) && node->get_input_size() > 2) {
                record_bias(node->get_input_node_shared_ptr(2));
            }
            // Do not inspect weights or preceding layers.
            continue;
        }
        if (ov::is_type<ov::op::v1::Add>(node)) {
            bool matched = false;
            for (size_t i = 0; i < 2; i++) {
                auto linear = skip_convert(node->get_input_node_shared_ptr(1 - i));
                if (is_linear(linear) && ov::is_type<ov::op::v0::Constant>(skip_convert(node->get_input_node_shared_ptr(i)))) {
                    record_bias(node->get_input_node_shared_ptr(i));
                    stack.push_back(linear);
                    matched = true;
                    break;
                }
            }
            // An arbitrary Add may combine positional/auxiliary computations, not projections.
            result.complete = result.complete && matched;
            continue;
        }
        if (ov::is_type<ov::op::v0::Constant>(node)) {
            continue;
        }
        if (ov::is_type<ov::op::v0::Concat>(node)) {
            for (const auto& input : node->input_values()) {
                stack.push_back(input.get_node_shared_ptr());
            }
            continue;
        }
        // These operations carry all K values of input 0 without selecting a subset of channels; other inputs
        // describe layout or RoPE trigonometric tables and must not be searched for biases.
        // Split/VariadicSplit/Slice/StridedSlice/Gather are deliberately not listed: they select a subset of
        // channels (e.g. K out of a fused QKV projection), and without tracking the output port and the static
        // channel mapping the bias of the whole projection would wrongly be attributed to K.
        bool is_plain_rope = false;
        if (const auto rope = ov::as_type_ptr<ov::op::internal::RoPE>(node)) {
            const auto& cfg = rope->get_config();
            // RoPE variants that slice the input or consume a fused QKV tensor are not plain pass-through.
            is_plain_rope = cfg.slice_start == 0 && cfg.slice_stop == 0 && !cfg.is_qwen && !cfg.is_chatglm;
        }
        if (ov::is_type<ov::op::v0::Convert>(node) || ov::is_type<ov::op::v1::Reshape>(node) ||
            ov::is_type<ov::op::v1::Transpose>(node) || ov::is_type<ov::op::v0::Squeeze>(node) ||
            ov::is_type<ov::op::v0::Unsqueeze>(node) || ov::is_type<ov::op::v1::Broadcast>(node) ||
            ov::is_type<ov::op::v3::Broadcast>(node) || is_plain_rope) {
            stack.push_back(node->get_input_node_shared_ptr(0));
            continue;
        }
        // Unsupported paths (including channel-selecting ops) are unknown, not evidence that K has a zero bias.
        result.complete = false;
    }
    return result;
}

// Empirical threshold for the CVS-190235 workaround, not an int8 limit: asymmetric quantization covers larger
// finite ranges at a coarser step, and a bias magnitude alone does not determine the per-row range.
constexpr float k_bias_limit = 128.f;

// The K-bias gate is a conservative static-heuristic fallback for CVS-190235, not a general outlier detector or a
// root-cause fix. It is limited to the configuration where the failure was observed: f16 compute, int8 KV-cache,
// non-IMMAD device and grouped-query attention. Returns the reason the gate does not apply, or an empty string
// when it does. Other configurations (e.g. f32 compute, which was verified to work with int8 KV-cache) keep the
// requested compression.
std::string get_k_bias_gate_skip_reason(ov::element::Type compression_dt,
                                        bool supports_immad,
                                        ov::element::Type compute_dt,
                                        const std::shared_ptr<ov::intel_gpu::op::IndirectSDPA>& sdpa) {
    if (!cldnn::one_of(compression_dt, {ov::element::i8, ov::element::u8})) {
        return "KV-cache compression precision is not int8";
    }
    if (supports_immad) {
        return "device supports IMMAD";
    }
    if (compute_dt != ov::element::f16) {
        return "compute precision is not f16";
    }

    auto heads_count = [](const ov::PartialShape& shape, std::vector<int64_t> order) -> ov::Dimension {
        if (shape.rank().is_dynamic()) {
            return ov::Dimension::dynamic();
        }
        if (order.size() != static_cast<size_t>(shape.size())) {
            order.resize(shape.size());
            std::iota(order.begin(), order.end(), 0);
        }
        return shape[order[1]];
    };
    const auto q_heads = heads_count(sdpa->get_input_partial_shape(0), sdpa->get_input0_transpose_order());
    const auto kv_heads = heads_count(sdpa->get_input_partial_shape(1), sdpa->get_input1_transpose_order());
    if (q_heads.is_dynamic() || kv_heads.is_dynamic() || q_heads.get_length() == kv_heads.get_length()) {
        return "attention is not statically known to use grouped-query heads";
    }
    return {};
}

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

        const auto& layer_name = sdpa_node->get_friendly_name();
        // Check the applicability scope first so that out-of-scope configurations do not pay for the bias analysis.
        const auto not_applicable_reason = get_k_bias_gate_skip_reason(compression_dt,
                                                                       supports_immad,
                                                                       query_node->get_output_element_type(0),
                                                                       sdpa_node);
        if (!not_applicable_reason.empty()) {
            GPU_DEBUG_LOG << "K bias gate is not applied for " << layer_name << ": " << not_applicable_reason
                          << ", compression is kept\n";
        } else {
            const auto k_bias = analyze_k_bias(key_new_token_node);
            if (k_bias.max_abs > k_bias_limit) {
                GPU_DEBUG_LOG << "Skipping KV-cache compression for " << layer_name << ": heuristic fallback, K projection max|bias|="
                              << k_bias.max_abs << " exceeds limit=" << k_bias_limit << ", original cache type is kept\n";
                return false;
            }
            if (k_bias.budget_exhausted) {
                GPU_DEBUG_LOG << "K bias analysis for " << layer_name << ": unique-node budget exhausted, compression is kept\n";
            } else if (!k_bias.complete) {
                GPU_DEBUG_LOG << "K bias analysis for " << layer_name << ": unsupported data path or non-constant bias, compression is kept\n";
            } else if (!k_bias.found) {
                GPU_DEBUG_LOG << "K bias analysis for " << layer_name << ": no projection bias found, compression is kept\n";
            } else {
                GPU_DEBUG_LOG << "K bias analysis for " << layer_name << ": maximum detected bias=" << k_bias.max_abs
                              << " <= limit=" << k_bias_limit << ", compression is kept\n";
            }
        }

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

KVCacheCompression::KVCacheCompression(ov::element::Type compression_dt, bool supports_immad) {
    add_matcher<ov::intel_gpu::KVCacheCompressionMatcher>(compression_dt, supports_immad);
}

}  // namespace ov::intel_gpu
