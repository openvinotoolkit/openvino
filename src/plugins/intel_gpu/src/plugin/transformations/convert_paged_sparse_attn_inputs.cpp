// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_paged_sparse_attn_inputs.hpp"

#include <optional>
#include <string>

#include "intel_gpu/op/sparse_kv_quant.hpp"
#include "intel_gpu/op/sparse_pa_config.hpp"
#include "op/qsa_op_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/paged_sparse_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/util/log.hpp"

namespace ov::intel_gpu {

namespace {

using PagedSparseAttention = ov::op::internal::PagedSparseAttention;
using In = op::SparsePAInputIdx;
using op::qsa_util::rt_size;

ov::pass::ConvertPagedAttnInputs::KVCacheConfig make_kv_cache_config(const ConvertPagedSparseAttnInputs::CacheConfig& c) {
    ov::pass::ConvertPagedAttnInputs::KVCacheConfig config;
    config.keyCachePrecision = c.precision;
    config.valueCachePrecision = c.precision;
    // Only used by ConvertPagedAttnInputs to turn f16 caches into bf16 ones, which the precision update below undoes.
    config.inferencePrecision = ov::element::f16;
    config.keyCacheBlockSize = c.page_size;
    config.valueCacheBlockSize = c.page_size;
    config.keyCacheQuantBychannel = c.key_by_channel;
    config.keyCacheGroupSize = c.key_by_channel ? op::paged_by_channel_group_size : 0;
    config.valueCacheQuantBychannel = false;
    config.valueCacheGroupSize = 0;
    // [blocks, num_kv_heads, rows, row width] for both caches: the rows hold the tokens, then the parameters.
    config.keyCacheDimOrder = {0, 1, 2, 3};
    config.valueCacheDimOrder = {0, 1, 2, 3};
    return config;
}

// A quantized cache is a u8 page [rows, row bytes] of kv_page_layout; an f16 cache keeps [page_size, head_size].
void update_shape(const ov::element::Type precision, const bool by_channel, const size_t /*group_num*/, int64_t& head_size, int64_t& block_size) {
    const auto quant = op::paged_cache_quant(precision, by_channel);
    if (!quant.enabled()) {
        return;
    }
    const auto layout = op::kv_page_layout(quant, static_cast<size_t>(block_size), static_cast<size_t>(head_size));
    head_size = static_cast<int64_t>(layout.row_bytes);
    block_size = static_cast<int64_t>(layout.total_rows);
}

void update_precision(ov::element::Type& precision) {
    precision = op::paged_cache_quant(precision, false).enabled() ? ov::element::u8 : ov::element::f16;
}

}  // namespace

ConvertPagedSparseAttnInputs::ConvertPagedSparseAttnInputs(const CacheConfig& cache_config)
    : ov::pass::ConvertPagedAttnInputs(make_kv_cache_config(cache_config), update_shape, update_precision) {}

bool ConvertPagedSparseAttnInputs::skip(const std::shared_ptr<const ov::Node>& node) {
    if (!ov::is_type<PagedSparseAttention>(node)) {
        return true;
    }
    for (const auto* key : {"num_k_heads", "k_head_size", "num_v_heads", "v_head_size"}) {
        if (node->get_rt_info().count(key) == 0) {
            // Caches without the head geometry are taken as they are (f16 [blocks, num_kv_heads, page_size, head_size]).
            return true;
        }
    }
    const auto key_cache = ov::as_type_ptr<const ov::op::v0::Parameter>(node->get_input_node_shared_ptr(In::KEY_CACHE));
    const auto value_cache = ov::as_type_ptr<const ov::op::v0::Parameter>(node->get_input_node_shared_ptr(In::VALUE_CACHE));
    return !key_cache || !value_cache || key_cache == value_cache || key_cache->get_output_target_inputs(0).size() != 1 ||
           value_cache->get_output_target_inputs(0).size() != 1;
}

ov::element::Type ConvertPagedSparseAttnInputs::select_precision(const ov::Model& model, const CacheConfig& cache_config) {
    const auto key_quant = op::paged_cache_quant(cache_config.precision, cache_config.key_by_channel);
    const auto value_quant = op::paged_cache_quant(cache_config.precision, false);
    if (!key_quant.enabled()) {
        return cache_config.precision;
    }
    for (const auto& node : model.get_ordered_ops()) {
        const auto attention = ov::as_type_ptr<PagedSparseAttention>(node);
        if (!attention || skip(attention)) {
            continue;
        }
        const auto k_head_size = rt_size(*attention, "k_head_size");
        const auto v_head_size = rt_size(*attention, "v_head_size");
        const auto key_block_size = static_cast<size_t>(attention->get_key_block_size());
        std::string reason;
        if (!op::validate_kv_quant(key_quant, true, cache_config.page_size, *k_head_size, key_block_size, reason) ||
            !op::validate_kv_quant(value_quant, false, cache_config.page_size, *v_head_size, key_block_size, reason)) {
            OPENVINO_WARN("[GPU] PagedSparseAttention '",
                          attention->get_friendly_name(),
                          "' cannot use a ",
                          cache_config.precision,
                          " KV cache (",
                          reason,
                          "): the sparse attention caches are kept in f16.");
            return ov::element::f16;
        }
    }
    return cache_config.precision;
}

}  // namespace ov::intel_gpu
