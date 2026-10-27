// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/sparse_pa.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/paged_sparse_attention.hpp"

namespace ov::intel_gpu {

static void CreatePagedSparseAttentionOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::PagedSparseAttention>& op) {
    using In = cldnn::sparse_pa::InputIdx;
    constexpr size_t scale_idx = 11;
    constexpr size_t sinks_idx = 12;
    constexpr size_t input_count = 13;

    validate_inputs_count(op, {input_count});
    const auto layer_name = layer_type_name_ID(op);

    // The GPU kernels need every head dimension statically known.
    const auto& query_ps = op->get_input_partial_shape(In::QUERY);
    const auto& key_cache_ps = op->get_input_partial_shape(In::KEY_CACHE);
    const auto& value_cache_ps = op->get_input_partial_shape(In::VALUE_CACHE);
    const auto& selected_ps = op->get_input_partial_shape(In::SELECTED_BLOCK_INDICES);
    OPENVINO_ASSERT(query_ps.rank().is_static() && key_cache_ps.rank().is_static() && value_cache_ps.rank().is_static() && selected_ps.rank().is_static() &&
                        query_ps[1].is_static() && key_cache_ps[1].is_static() && key_cache_ps[3].is_static() && value_cache_ps[3].is_static() &&
                        selected_ps[1].is_static(),
                    "[GPU] PagedSparseAttention '",
                    layer_name,
                    "': head counts and head sizes must be static");

    cldnn::sparse_pa::Config config;
    config.num_kv_heads = static_cast<size_t>(key_cache_ps[1].get_length());
    config.k_head_size = static_cast<size_t>(key_cache_ps[3].get_length());
    config.v_head_size = static_cast<size_t>(value_cache_ps[3].get_length());
    config.num_heads = static_cast<size_t>(query_ps[1].get_length()) / config.k_head_size;
    config.num_selection_heads = static_cast<size_t>(selected_ps[1].get_length());
    config.key_block_size = static_cast<size_t>(op->get_key_block_size());

    std::string reason;
    OPENVINO_ASSERT(cldnn::sparse_pa::validate_config(config, reason), "[GPU] Cannot lower PagedSparseAttention op '", layer_name, "': ", reason);
    OPENVINO_ASSERT(config.num_selection_heads == 1,
                    "[GPU] PagedSparseAttention '",
                    layer_name,
                    "': only num_selection_heads == 1 is supported, got ",
                    config.num_selection_heads);
    for (size_t idx : {In::QUERY, In::KEY, In::VALUE, In::KEY_CACHE, In::VALUE_CACHE}) {
        OPENVINO_ASSERT(op->get_input_element_type(idx) == ov::element::f16,
                        "[GPU] PagedSparseAttention '",
                        layer_name,
                        "': input ",
                        idx,
                        " must be f16, got ",
                        op->get_input_element_type(idx));
    }

    // Attention sinks are not supported by the GPU kernels (explicit rejection, not a silent fallback).
    const auto& sinks_ps = op->get_input_partial_shape(sinks_idx);
    const bool sinks_empty = sinks_ps.rank().is_static() && sinks_ps.size() == 1 && sinks_ps[0].is_static() && sinks_ps[0].get_length() == 0;
    OPENVINO_ASSERT(sinks_empty, "[GPU] PagedSparseAttention '", layer_name, "': the sinks input is not supported by the GPU implementation");

    // The scale input is folded into the primitive, so it must be a Constant.
    auto scale_const = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(scale_idx));
    OPENVINO_ASSERT(scale_const != nullptr, "[GPU] PagedSparseAttention '", layer_name, "': scale input must be a Constant in this lowering pass");
    const float scale = scale_const->cast_vector<float>().at(0);

    // Only QUERY..SELECTED_BLOCK_COUNT are consumed by the kernels; scale and sinks are not primitive inputs.
    const auto full_inputs = p.GetInputInfo(op);
    const std::vector<cldnn::input_info> inputs(full_inputs.begin(), full_inputs.begin() + In::COUNT);

    cldnn::sparse_pa prim(layer_name, inputs, config, scale);
    prim.num_outputs = op->get_output_size();
    prim.output_data_types = get_output_data_types(op);
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, PagedSparseAttention);

}  // namespace ov::intel_gpu
