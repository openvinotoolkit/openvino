// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kv_cache_compression.hpp"

#include <memory>
#include <numeric>
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
// Zero point type of an asymmetric compressed KV cache, shared by all its producers: an integer one on immad devices (the
// planar layout), except for 4-bit caches. u4 has only 16 levels, so rounding the zero point to an integer would add a
// systematic error of (zp - round(zp)) * scale to every element.
element::Type kv_cache_zp_type(bool supports_immad, size_t bits, element::Type scale_dt) {
    return supports_immad && bits != 4 ? element::i8 : scale_dt;
}

std::vector<ov::op::util::VariableInfo> get_variable_infos(const ov::op::util::VariableInfo& data_variable_info,
                                                           const ov::op::internal::DynamicQuantize::Attributes& quantization_attrs) {
    std::vector<ov::op::util::VariableInfo> infos;

    // Add initial data variable info
    infos.push_back(data_variable_info);

    const auto variable_id = data_variable_info.variable_id;
    const auto scale_dt = quantization_attrs.scale_dt;

    // Infer DQ shapes
    ov::op::internal::DynamicQuantize dq;
    dq.set_attrs(quantization_attrs);

    auto dq_shapes = ov::op::internal::DynamicQuantize::shape_infer(&dq, {data_variable_info.data_shape});
    const auto scale_shape = dq_shapes[1];

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

std::shared_ptr<ov::intel_gpu::op::ReadValues> update_past_read_value(std::shared_ptr<ov::intel_gpu::op::ReadValue> past_rv_node,
                                                                      const ov::op::internal::DynamicQuantize::Attributes& quantization_attrs) {
    auto variable = past_rv_node->get_variable();
    variable->update_data_type(quantization_attrs.quantization_dt);

    auto variable_infos = get_variable_infos(past_rv_node->get_variable()->get_info(), quantization_attrs);
    auto new_past_rv_node = std::make_shared<ov::intel_gpu::op::ReadValues>();

    if (past_rv_node->get_input_size() == 0) {
        new_past_rv_node = std::make_shared<ov::intel_gpu::op::ReadValues>(past_rv_node->get_variable(), variable_infos);
    } else {
        auto initializer_dq = std::make_shared<ov::op::internal::DynamicQuantize>(past_rv_node->input_value(0), quantization_attrs);
        initializer_dq->set_friendly_name(past_rv_node->input_value(0).get_node_shared_ptr()->get_friendly_name() + "_dyn_quan");
        ov::copy_runtime_info(past_rv_node->input_value(0).get_node_shared_ptr(), initializer_dq);

        OutputVector initializer_outputs = {initializer_dq->output(0), initializer_dq->output(1)};

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

std::shared_ptr<ov::intel_gpu::op::KVCacheCompressed> update_kv_cache(std::shared_ptr<ov::intel_gpu::op::ReadValue> past_rv_node,
                                                                      std::shared_ptr<ov::intel_gpu::op::KVCache> kv_cache_node,
                                                                      const ov::Output<ov::Node>& beam_idx,
                                                                      const ov::op::internal::DynamicQuantize::Attributes& quantization_attrs) {
    OutputVector kv_cache_inputs = {past_rv_node->output(0), kv_cache_node->input_value(1), beam_idx};
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

// Pattern of a plain KVCache feeding SparseSDPA exclusively: `KVCache(ReadValue, new)` with a static batch (one sequence per
// batch row), or `KVCache(Gather(ReadValue, beam_idx), new)`.
struct SparseCachePattern {
    std::shared_ptr<ov::Node> cache;
    std::shared_ptr<ov::Node> past;           // ReadValue of the single-sequence form
    std::shared_ptr<ov::Node> gathered_past;  // ReadValue behind the Gather
    std::shared_ptr<ov::Node> gather;
    std::shared_ptr<ov::Node> beam_idx;
};

SparseCachePattern make_sparse_cache_pattern() {
    using namespace ov::pass::pattern;
    using ov::pass::pattern::op::Predicate;

    // Exclusive consumers: no other node may read the f16 state or cache that the rewrite retypes.
    const auto f16_rank4_state = consumers_count(1) && type_matches(element::f16) && rank_equals(4);  // [B, Hkv, S, D]
    const auto static_batch = Predicate([](const ov::Output<ov::Node>& out) {
        const auto& ps = out.get_partial_shape();
        return ps.rank().is_static() && ps[0].is_static() && ps[0].get_length() >= 1;
    });
    const auto plain_cache = Predicate([](const ov::Output<ov::Node>& out) {
        auto kv_cache = ov::as_type_ptr<ov::intel_gpu::op::KVCache>(out.get_node_shared_ptr());
        return kv_cache && !ov::is_type<ov::intel_gpu::op::KVCacheCompressed>(kv_cache) && !kv_cache->get_indirect() && !kv_cache->get_trim() &&
               !kv_cache->get_update_kv() && ov::util::normalize(kv_cache->get_concat_axis(), 4) == 2;
    });
    const auto batch_gather = Predicate([](const ov::Output<ov::Node>& out) {
        auto gather = ov::as_type_ptr<ov::op::v8::Gather>(out.get_node_shared_ptr());
        return gather && gather->get_batch_dims() == 0;
    });

    SparseCachePattern p;
    auto new_token = any_input();
    p.past = wrap_type<ov::intel_gpu::op::ReadValue>(f16_rank4_state && static_batch);
    p.gathered_past = wrap_type<ov::intel_gpu::op::ReadValue>(f16_rank4_state);
    p.beam_idx = wrap_type<ov::op::v0::Parameter>();
    const auto axis_zero = wrap_type<ov::op::v0::Constant>(ov::op::util::constant_predicate<int64_t>([](const std::vector<int64_t>& value) {
        return value.size() == 1 && value[0] == 0;
    }));
    p.gather = wrap_type<ov::op::v8::Gather>({p.gathered_past, p.beam_idx, axis_zero}, consumers_count(1) && batch_gather);
    const auto cache_pred = f16_rank4_state && plain_cache;
    auto single_cache = wrap_type<ov::intel_gpu::op::KVCache>({p.past, new_token}, cache_pred);
    auto gathered_cache = wrap_type<ov::intel_gpu::op::KVCache>({p.gather, new_token}, cache_pred);
    p.cache = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{single_cache, gathered_cache});
    return p;
}

// A KVCache feeding SparseSDPA, reduced to what the compression needs.
struct SparseSDPACache {
    std::shared_ptr<ov::intel_gpu::op::KVCache> kv_cache;
    std::shared_ptr<ov::intel_gpu::op::ReadValue> past;
    ov::Output<ov::Node> beam_idx;
};

SparseSDPACache extract_sparse_sdpa_cache(const ov::pass::pattern::PatternValueMap& pattern_map, const SparseCachePattern& p) {
    SparseSDPACache res;
    res.kv_cache = ov::as_type_ptr<ov::intel_gpu::op::KVCache>(pattern_map.at(p.cache).get_node_shared_ptr());
    if (pattern_map.count(p.gather)) {
        res.past = ov::as_type_ptr<ov::intel_gpu::op::ReadValue>(pattern_map.at(p.gathered_past).get_node_shared_ptr());
        res.beam_idx = pattern_map.at(p.beam_idx);
    } else {
        // Without a gather every batch row keeps its own cache, which an identity beam index expresses.
        res.past = ov::as_type_ptr<ov::intel_gpu::op::ReadValue>(pattern_map.at(p.past).get_node_shared_ptr());
        std::vector<int32_t> identity(static_cast<size_t>(res.past->get_output_partial_shape(0)[0].get_length()));
        std::iota(identity.begin(), identity.end(), 0);
        res.beam_idx = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{identity.size()}, identity);
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

    auto sdpa_without_attn_mask_m = wrap_type<ov::intel_gpu::op::IndirectSDPA>({query, key_cache, value_cache, input_beam_table});
    auto sdpa_with_attn_mask_m = wrap_type<ov::intel_gpu::op::IndirectSDPA>({query, key_cache, value_cache, input_attn_mask, input_beam_table});
    auto sdpa_with_attn_mask_and_scale_m =
        wrap_type<ov::intel_gpu::op::IndirectSDPA>({query, key_cache, value_cache, input_attn_mask, input_scale, input_beam_table});

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
            config.zp_dt = kv_cache_zp_type(supports_immad, ov::element::Type(compression_dt).bitwidth(), scale_dt);
        }

        key_past_rv_node = update_past_read_value(key_past_rv_node, config);
        value_past_rv_node = update_past_read_value(value_past_rv_node, config);

        auto new_key_cache = update_kv_cache(key_past_rv_node, key_cache_node, key_cache_node->input_value(2), config);
        auto new_value_cache = update_kv_cache(value_past_rv_node, value_cache_node, value_cache_node->input_value(2), config);

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

// SparseSDPA reads K/V as dense [B, Hkv, S, D] states, so its caches are compressed per token with planar scales / zero
// points: the KV cache precision selects the code width (8/4-bit; f16 caches stay uncompressed) and the zero point type
// (see kv_cache_zp_type), as for the stateful SDPA (KVCacheCompressionMatcher). Unlike that matcher, it needs the planar
// (immad) layout: without immad the caches stay uncompressed instead of using interleaved scales / zero points.
//  - Per token is required: SparseSDPA::validate_config rejects any other mode for a dense cache (BY_CHANNEL keys are a
//    paged-cache feature).
//  - Asymmetric is the default choice here, mirroring KVCacheCompressionMatcher, which always asks for it. It is a policy,
//    not a limit of the op: SparseSDPA also accepts symmetric dense caches (no zero point inputs). This pass takes no
//    quantization parameter, so a producer that needs symmetric caches has to add one.
//  - Beam reordering is not supported yet: SparseSDPA reads no beam table, so the beam table output of the compressed
//    KVCache stays unused. The compressed KVCache does not physically reorder the past (it only records beam_idx in that
//    table), so the `Gather(ReadValue, beam_idx)` form is only correct while beam_idx is the identity. Nothing checks
//    this at runtime.
class SparseSDPAKVCacheCompressionMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SparseSDPAKVCacheCompressionMatcher");
    SparseSDPAKVCacheCompressionMatcher(ov::element::Type compression_dt, bool supports_immad);
};

SparseSDPAKVCacheCompressionMatcher::SparseSDPAKVCacheCompressionMatcher(ov::element::Type compression_dt, bool supports_immad) {
    using namespace ov::pass::pattern;
    using In = ov::intel_gpu::op::SparseSDPA::InputIdx;
    using ov::intel_gpu::op::KVQuant;

    // The cache kernels pack the codes at the width of the global KV cache precision (4-bit rows hold two codes per byte).
    if (!supports_immad || !cldnn::one_of(compression_dt, {element::i8, element::u8, element::i4, element::u4})) {
        return;
    }
    const auto quant = KVQuant{KVQuant::Mode::BY_TOKEN, static_cast<uint8_t>(compression_dt.bitwidth()), false, 0};
    ov::op::internal::DynamicQuantize::Attributes attrs;
    attrs.quantization_type = ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;
    attrs.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::Planar;
    attrs.group_sizes = {1, 1, 1, UINT64_MAX};  // one scale / zero point per token row
    attrs.quantization_dt = element::i8;
    attrs.scale_dt = element::f16;
    attrs.zp_dt = kv_cache_zp_type(supports_immad, quant.bits, attrs.scale_dt);
    attrs.scales_zp_output_order = {0, 1, 2, 3};

    const auto key_cache = make_sparse_cache_pattern();
    const auto value_cache = make_sparse_cache_pattern();
    // Uncompressed SparseSDPA without sinks (6 core inputs, plus the optional scale) whose compressed configuration is valid.
    const auto compressible_sdpa = ov::pass::pattern::op::Predicate([quant](const ov::Output<ov::Node>& out) {
        auto sdpa = ov::as_type_ptr<ov::intel_gpu::op::SparseSDPA>(out.get_node_shared_ptr());
        if (!sdpa || sdpa->get_config().kv_compressed()) {
            return false;
        }
        auto compressed_config = sdpa->get_config();
        compressed_config.key_quant = quant;
        compressed_config.value_quant = quant;
        std::string reason;
        return ov::intel_gpu::op::SparseSDPA::validate_config(compressed_config, reason);
    });
    auto query = any_input(type_matches(element::f16));
    auto selected_indices = any_input();
    auto selected_count = any_input();
    auto key_length = any_input();
    auto scale = any_input();
    auto sdpa_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{
        wrap_type<ov::intel_gpu::op::SparseSDPA>({query, key_cache.cache, value_cache.cache, selected_indices, selected_count, key_length}, compressible_sdpa),
        wrap_type<ov::intel_gpu::op::SparseSDPA>({query, key_cache.cache, value_cache.cache, selected_indices, selected_count, key_length, scale},
                                                 compressible_sdpa)});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }
        const auto& pattern_map = m.get_pattern_value_map();
        auto sdpa = ov::as_type_ptr<ov::intel_gpu::op::SparseSDPA>(m.get_match_root());
        const auto& sdpa_config = sdpa->get_config();

        auto compressed_config = sdpa_config;
        compressed_config.key_quant = quant;
        compressed_config.value_quant = quant;

        auto key = extract_sparse_sdpa_cache(pattern_map, key_cache);
        auto value = extract_sparse_sdpa_cache(pattern_map, value_cache);

        auto key_rv = update_past_read_value(key.past, attrs);
        auto value_rv = update_past_read_value(value.past, attrs);
        auto new_key_cache = update_kv_cache(key_rv, key.kv_cache, key.beam_idx, attrs);
        auto new_value_cache = update_kv_cache(value_rv, value.kv_cache, value.beam_idx, attrs);

        ov::OutputVector args(compressed_config.scale_input_idx());
        for (size_t i : {In::QUERY, In::SELECTED_BLOCK_INDICES, In::SELECTED_BLOCK_COUNT, In::KEY_LENGTH}) {
            args[i] = sdpa->input_value(i);
        }
        args[In::KEY] = new_key_cache->output(0);
        args[In::VALUE] = new_value_cache->output(0);
        args[In::KEY_SCALE] = new_key_cache->output(2);
        args[In::VALUE_SCALE] = new_value_cache->output(2);
        args[In::KEY_ZP] = new_key_cache->output(3);
        args[In::VALUE_ZP] = new_value_cache->output(3);
        if (sdpa->has_scale_input()) {
            args.push_back(sdpa->input_value(sdpa_config.scale_input_idx()));
        }

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
