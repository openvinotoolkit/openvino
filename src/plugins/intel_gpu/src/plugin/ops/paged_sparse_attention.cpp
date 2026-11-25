// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_sparse_attention.hpp"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "intel_gpu/op/sparse_kv_quant.hpp"
#include "intel_gpu/op/sparse_pa_config.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
#include "intel_gpu/primitives/sparse_pa.hpp"
#include "openvino/op/constant.hpp"
#include "plugin/transformations/op/qsa_op_utils.hpp"

// Lowers ov::op::internal::PagedSparseAttention to cldnn::sparse_pa, like CreatePagedAttentionExtensionOp lowers
// PagedAttentionExtension:
//  - the head geometry comes from the `num_k_heads` / `k_head_size` / `num_v_heads` / `v_head_size` runtime info the
//    paged-attention conversion stamps on the op, or else from static cache and projection shapes (num_kv_heads is
//    key_cache dim 1, the head sizes are the key / value widths over it, num_heads the query width over the key head
//    size);
//  - the cache quantization comes from the cache element type and the KV cache options: u8 caches are quantized as
//    ConvertPagedSparseAttnInputs laid them out (see sparse_kv_quant.hpp), f16 caches are not;
//  - the scale must be a Constant (folded into the primitive), sinks are not supported.

namespace ov::intel_gpu {

namespace {

using SparsePAConfig = ov::intel_gpu::op::SparsePAConfig;
using In = ov::intel_gpu::op::SparsePAInputIdx;
using ov::intel_gpu::op::qsa_util::rt_size;

std::optional<size_t> static_dim(const ov::PartialShape& shape, int64_t rank, size_t axis) {
    if (shape.rank().is_dynamic() || shape.rank().get_length() != rank || shape[axis].is_dynamic()) {
        return std::nullopt;
    }
    return static_cast<size_t>(shape[axis].get_length());
}

// The head geometry of `op`, see the file description.
bool derive_heads(const ov::op::internal::PagedSparseAttention& op, SparsePAConfig& config, std::string& reason) {
    const auto query_width = static_dim(op.get_input_partial_shape(In::QUERY), 2, 1);
    const auto key_width = static_dim(op.get_input_partial_shape(In::KEY), 2, 1);
    const auto value_width = static_dim(op.get_input_partial_shape(In::VALUE), 2, 1);
    const auto selection_heads = static_dim(op.get_input_partial_shape(In::SELECTED_BLOCK_INDICES), 3, 1);
    if (!query_width || !key_width || !value_width || !selection_heads) {
        reason = "the query / key / value widths and the selection head count must be static";
        return false;
    }
    const auto k_heads = rt_size(op, "num_k_heads");
    const auto k_head_size = rt_size(op, "k_head_size");
    const auto v_heads = rt_size(op, "num_v_heads");
    const auto v_head_size = rt_size(op, "v_head_size");
    if (k_heads && k_head_size && v_heads && v_head_size) {
        if (*k_heads != *v_heads) {
            reason = "the key and value head counts differ";
            return false;
        }
        config.num_kv_heads = *k_heads;
        config.k_head_size = *k_head_size;
        config.v_head_size = *v_head_size;
    } else {
        const auto kv_heads = static_dim(op.get_input_partial_shape(In::KEY_CACHE), 4, 1);
        if (!kv_heads || *kv_heads == 0 || *key_width % *kv_heads != 0 || *value_width % *kv_heads != 0) {
            reason = "without the head geometry runtime info the key cache head count must be static and divide the key / value widths";
            return false;
        }
        config.num_kv_heads = *kv_heads;
        config.k_head_size = *key_width / *kv_heads;
        config.v_head_size = *value_width / *kv_heads;
    }
    if (config.k_head_size == 0 || *query_width % config.k_head_size != 0 || *key_width != config.num_kv_heads * config.k_head_size ||
        *value_width != config.num_kv_heads * config.v_head_size) {
        reason = "the query / key / value widths do not match the head geometry";
        return false;
    }
    config.num_heads = *query_width / config.k_head_size;
    config.num_selection_heads = *selection_heads;
    return true;
}

// Checks the cache shape against the config: [blocks, Hkv, page_size, head_size] f16, or u8 kv_page_layout pages.
bool cache_matches(const ov::PartialShape& ps, const SparsePAConfig& config, const op::KVQuant& quant, size_t head_size) {
    if (ps.rank().is_dynamic()) {
        return true;
    }
    if (ps.size() != 4 || !ps[1].compatible(static_cast<int64_t>(config.num_kv_heads))) {
        return false;
    }
    if (!quant.enabled()) {
        return ps[3].compatible(static_cast<int64_t>(head_size)) && (config.page_size == 0 || ps[2].compatible(static_cast<int64_t>(config.page_size)));
    }
    const auto layout = op::kv_page_layout(quant, config.page_size, head_size);
    return ps[2].compatible(static_cast<int64_t>(layout.total_rows)) && ps[3].compatible(static_cast<int64_t>(layout.row_bytes));
}

bool has_sinks(const ov::Node& op) {
    const auto& ps = op.get_input_partial_shape(In::SINKS);
    return !ps.rank().is_static() || ps.size() != 1 || !ps[0].is_static() || ps[0].get_length() != 0;
}

}  // namespace

static void CreatePagedSparseAttentionOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::PagedSparseAttention>& op) {
    validate_inputs_count(op, {In::COUNT});
    const auto layer_name = layer_type_name_ID(op);

    SparsePAConfig config;
    std::string reason;
    OPENVINO_ASSERT(derive_heads(*op, config, reason), "[GPU] Cannot lower PagedSparseAttention '", layer_name, "': ", reason);
    config.key_block_size = static_cast<size_t>(op->get_key_block_size());

    // Quantized caches are the u8 pages ConvertPagedSparseAttnInputs lays out with the configured KV cache options.
    const auto& exec_config = p.get_config();
    const bool key_by_channel = exec_config.get_key_cache_quant_mode() == ov::internal::CacheQuantMode::BY_CHANNEL;
    for (auto [idx, quant] : {std::pair{In::KEY_CACHE, &config.key_quant}, std::pair{In::VALUE_CACHE, &config.value_quant}}) {
        const auto& type = op->get_input_element_type(idx);
        if (type == ov::element::u8) {
            *quant = op::paged_cache_quant(exec_config.get_kv_cache_precision(), idx == In::KEY_CACHE && key_by_channel);
            OPENVINO_ASSERT(quant->enabled(),
                            "[GPU] PagedSparseAttention '",
                            layer_name,
                            "': a u8 cache needs an 8- or 4-bit KV cache precision, got ",
                            exec_config.get_kv_cache_precision());
        } else {
            OPENVINO_ASSERT(type == ov::element::f16, "[GPU] PagedSparseAttention '", layer_name, "': input ", idx, " must be f16 or u8, got ", type);
        }
    }
    if (config.kv_quantized()) {
        config.page_size = cldnn::paged_attention::block_size;
    } else if (const auto page_size = static_dim(op->get_input_partial_shape(In::KEY_CACHE), 4, 2)) {
        config.page_size = *page_size;
    }

    OPENVINO_ASSERT(config.validate(reason), "[GPU] Cannot lower PagedSparseAttention '", layer_name, "': ", reason);
    OPENVINO_ASSERT(config.num_selection_heads == 1,
                    "[GPU] PagedSparseAttention '",
                    layer_name,
                    "': only num_selection_heads == 1 is supported, got ",
                    config.num_selection_heads);
    OPENVINO_ASSERT(cache_matches(op->get_input_partial_shape(In::KEY_CACHE), config, config.key_quant, config.k_head_size) &&
                        cache_matches(op->get_input_partial_shape(In::VALUE_CACHE), config, config.value_quant, config.v_head_size),
                    "[GPU] PagedSparseAttention '",
                    layer_name,
                    "': the caches must be [blocks, num_kv_heads, page_size, head_size] (quantized: [blocks, num_kv_heads, page_size + parameter "
                    "rows, bytes per row]), got ",
                    op->get_input_partial_shape(In::KEY_CACHE),
                    " and ",
                    op->get_input_partial_shape(In::VALUE_CACHE));
    for (size_t idx : {In::QUERY, In::KEY, In::VALUE}) {
        OPENVINO_ASSERT(op->get_input_element_type(idx) == ov::element::f16,
                        "[GPU] PagedSparseAttention '",
                        layer_name,
                        "': input ",
                        idx,
                        " must be f16, got ",
                        op->get_input_element_type(idx));
    }

    // Attention sinks are not supported by the GPU kernels (explicit rejection, not a silent fallback).
    OPENVINO_ASSERT(!has_sinks(*op), "[GPU] PagedSparseAttention '", layer_name, "': the sinks input is not supported by the GPU implementation");

    // The scale input is folded into the primitive, so it must be a Constant.
    const auto scale_const = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(In::SCALE));
    OPENVINO_ASSERT(scale_const != nullptr && ov::shape_size(scale_const->get_shape()) == 1,
                    "[GPU] PagedSparseAttention '",
                    layer_name,
                    "': the scale input must be a one-element Constant");
    const float scale = scale_const->cast_vector<float>()[0];

    // Only QUERY..SELECTED_BLOCK_COUNT are consumed by the kernels; scale and sinks are not primitive inputs.
    const auto full_inputs = p.GetInputInfo(op);
    const std::vector<cldnn::input_info> inputs(full_inputs.begin(), full_inputs.begin() + cldnn::sparse_pa::InputIdx::COUNT);

    cldnn::sparse_pa prim(layer_name, inputs, config, scale);
    prim.num_outputs = op->get_output_size();
    prim.output_data_types = get_output_data_types(op);
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, PagedSparseAttention);

}  // namespace ov::intel_gpu
