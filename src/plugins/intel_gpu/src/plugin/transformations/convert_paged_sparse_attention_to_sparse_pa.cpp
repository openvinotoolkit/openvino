// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_paged_sparse_attention_to_sparse_pa.hpp"

#include <memory>
#include <optional>
#include <string>

#include "intel_gpu/op/sparse_pa.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/paged_sparse_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"

namespace ov::intel_gpu {

namespace {

using SparsePA = ov::intel_gpu::op::SparsePA;
using In = SparsePA::InputIdx;

std::optional<size_t> static_dim(const ov::PartialShape& shape, int64_t rank, size_t axis) {
    if (shape.rank().is_dynamic() || shape.rank().get_length() != rank || shape[axis].is_dynamic()) {
        return std::nullopt;
    }
    return static_cast<size_t>(shape[axis].get_length());
}

std::optional<size_t> rt_size(const ov::Node& node, const std::string& key) {
    const auto& rt = node.get_rt_info();
    const auto it = rt.find(key);
    if (it == rt.end()) {
        return std::nullopt;
    }
    return it->second.as<size_t>();
}

// The head geometry of `op`, see the class description.
bool derive_heads(const ov::op::internal::PagedSparseAttention& op, SparsePA::Config& config) {
    const auto query_width = static_dim(op.get_input_partial_shape(In::QUERY), 2, 1);
    const auto key_width = static_dim(op.get_input_partial_shape(In::KEY), 2, 1);
    const auto value_width = static_dim(op.get_input_partial_shape(In::VALUE), 2, 1);
    const auto selection_heads = static_dim(op.get_input_partial_shape(In::SELECTED_BLOCK_INDICES), 3, 1);
    if (!query_width || !key_width || !value_width || !selection_heads) {
        return false;
    }
    const auto k_heads = rt_size(op, "num_k_heads");
    const auto k_head_size = rt_size(op, "k_head_size");
    const auto v_heads = rt_size(op, "num_v_heads");
    const auto v_head_size = rt_size(op, "v_head_size");
    if (k_heads && k_head_size && v_heads && v_head_size) {
        if (*k_heads != *v_heads) {
            return false;
        }
        config.num_kv_heads = *k_heads;
        config.k_head_size = *k_head_size;
        config.v_head_size = *v_head_size;
    } else {
        const auto kv_heads = static_dim(op.get_input_partial_shape(In::KEY_CACHE), 4, 1);
        if (!kv_heads || *kv_heads == 0 || *key_width % *kv_heads != 0 || *value_width % *kv_heads != 0) {
            return false;
        }
        config.num_kv_heads = *kv_heads;
        config.k_head_size = *key_width / *kv_heads;
        config.v_head_size = *value_width / *kv_heads;
    }
    if (config.k_head_size == 0 || *query_width % config.k_head_size != 0 || *key_width != config.num_kv_heads * config.k_head_size ||
        *value_width != config.num_kv_heads * config.v_head_size) {
        return false;
    }
    config.num_heads = *query_width / config.k_head_size;
    config.num_selection_heads = *selection_heads;
    return true;
}

// The SparsePA quantization of a cache stored with `precision`.
op::KVQuant cache_quant(const ov::element::Type& precision, bool by_channel) {
    op::KVQuant quant;
    if (precision == ov::element::i8 || precision == ov::element::u8) {
        quant.bits = 8;
    } else if (precision == ov::element::i4 || precision == ov::element::u4) {
        quant.bits = 4;
    } else {
        return quant;
    }
    quant.mode = by_channel ? op::KVQuant::Mode::BY_CHANNEL : op::KVQuant::Mode::BY_TOKEN;
    quant.group_size = by_channel ? 16 : 0;
    return quant;
}

void set_cache_layout(const std::shared_ptr<ov::op::v0::Parameter>& cache, const op::KVQuant& quant, size_t num_kv_heads, size_t page_size, size_t head_size) {
    const auto heads = static_cast<int64_t>(num_kv_heads);
    if (quant.enabled()) {
        const auto layout = op::kv_page_layout(quant, page_size, head_size);
        cache->set_element_type(ov::element::u8);
        cache->set_partial_shape({-1, heads, static_cast<int64_t>(layout.total_rows), static_cast<int64_t>(layout.row_bytes)});
    } else {
        cache->set_element_type(ov::element::f16);
        cache->set_partial_shape({-1, heads, static_cast<int64_t>(page_size), static_cast<int64_t>(head_size)});
    }
    ov::enable_keep_const_precision(cache);
    cache->validate_and_infer_types();
}

}  // namespace

ConvertPagedSparseAttentionToSparsePA::ConvertPagedSparseAttentionToSparsePA(const CacheConfig& cache_config) {
    using namespace ov::pass::pattern;

    const auto p_attention = wrap_type<ov::op::internal::PagedSparseAttention>();

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto attention = ov::as_type_ptr<ov::op::internal::PagedSparseAttention>(m.get_match_root());
        // SparsePA derives from PagedSparseAttention: it is matched too, but already lowered.
        if (!attention || ov::is_type<SparsePA>(attention) || transformation_callback(attention)) {
            return false;
        }

        SparsePA::Config config;
        if (!derive_heads(*attention, config)) {
            return false;
        }
        config.key_block_size = static_cast<size_t>(attention->get_key_block_size());

        const auto key_cache = ov::as_type_ptr<ov::op::v0::Parameter>(attention->get_input_node_shared_ptr(In::KEY_CACHE));
        const auto value_cache = ov::as_type_ptr<ov::op::v0::Parameter>(attention->get_input_node_shared_ptr(In::VALUE_CACHE));
        const bool owns_caches = key_cache && value_cache && key_cache != value_cache && key_cache->get_output_target_inputs(0).size() == 1 &&
                                 value_cache->get_output_target_inputs(0).size() == 1;
        std::string reason;
        if (owns_caches) {
            if (const auto declared_page_size = static_dim(key_cache->get_partial_shape(), 4, 2)) {
                config.page_size = *declared_page_size;
            } else {
                config.page_size = cache_config.page_size;
                config.key_quant = cache_quant(cache_config.precision, cache_config.key_by_channel);
                config.value_quant = cache_quant(cache_config.precision, false);
                if (config.kv_quantized() && !SparsePA::validate_config(config, reason)) {
                    // SparsePA cannot quantize this geometry: keep the caches in f16.
                    config.key_quant = config.value_quant = op::KVQuant{};
                }
            }
        } else {
            // Caches computed in the graph are taken as they are, with the page size of their shape.
            for (size_t idx : {In::KEY_CACHE, In::VALUE_CACHE}) {
                const auto& type = attention->get_input_element_type(idx);
                if (type.is_static() && type != ov::element::f16) {
                    return false;
                }
            }
        }
        if (!SparsePA::validate_config(config, reason)) {
            return false;
        }

        if (owns_caches) {
            set_cache_layout(key_cache, config.key_quant, config.num_kv_heads, config.page_size, config.k_head_size);
            set_cache_layout(value_cache, config.value_quant, config.num_kv_heads, config.page_size, config.v_head_size);
        }
        const auto sparse_pa = std::make_shared<SparsePA>(attention->input_values(), config);
        sparse_pa->set_friendly_name(attention->get_friendly_name());
        ov::copy_runtime_info(attention, sparse_pa);
        ov::replace_node(attention, sparse_pa);
        return true;
    };

    register_matcher(std::make_shared<Matcher>(p_attention, "ConvertPagedSparseAttentionToSparsePA"), callback);
}

}  // namespace ov::intel_gpu
