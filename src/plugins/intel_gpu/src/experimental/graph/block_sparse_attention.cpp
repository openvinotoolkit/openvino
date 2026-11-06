// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sstream>
#include <string>
#include <vector>

#include "block_sparse_attention_inst.h"
#include "json_object.h"
#include "primitive_type_base.h"
#include "to_string_utils.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(block_sparse_attention)

layout block_sparse_attention_inst::calc_output_layout(const block_sparse_attention_node& node, const kernel_impl_params& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
std::vector<layout> block_sparse_attention_inst::calc_output_layouts(const block_sparse_attention_node& /*node*/, const kernel_impl_params& impl_param) {
    OPENVINO_ASSERT(impl_param.input_layouts.size() == 4 || impl_param.input_layouts.size() == 5, "block_sparse_attention must have 4 or 5 inputs");

    const auto& q_layout = impl_param.get_input_layout(BlockSparseAttentionInputIdx::QUERY);
    const auto& v_layout = impl_param.get_input_layout(BlockSparseAttentionInputIdx::VALUE);

    const auto& q_shape = q_layout.get_partial_shape();  // [B, Lq, Hq, E]
    const auto& v_shape = v_layout.get_partial_shape();  // [B, Lk, Hkv, Ev]

    const auto b_dim = q_shape.rank().is_static() ? q_shape[0] : ov::Dimension::dynamic();
    const auto lq_dim = q_shape.rank().is_static() ? q_shape[1] : ov::Dimension::dynamic();
    const auto hq_dim = q_shape.rank().is_static() ? q_shape[2] : ov::Dimension::dynamic();
    const auto ev_dim = v_shape.rank().is_static() ? v_shape[3] : ov::Dimension::dynamic();

    const ov::PartialShape output_shape{b_dim, lq_dim, hq_dim, ev_dim};
    return {layout(output_shape, q_layout.data_type, format::bfyx)};
}

template std::vector<layout> block_sparse_attention_inst::calc_output_layouts<ov::PartialShape>(const block_sparse_attention_node& node,
                                                                                                const kernel_impl_params& impl_param);

std::string block_sparse_attention_inst::to_string(const block_sparse_attention_node& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite block_sparse_attention_info;
    block_sparse_attention_info.add("query", node.input(BlockSparseAttentionInputIdx::QUERY).id());
    block_sparse_attention_info.add("key", node.input(BlockSparseAttentionInputIdx::KEY).id());
    block_sparse_attention_info.add("value", node.input(BlockSparseAttentionInputIdx::VALUE).id());
    block_sparse_attention_info.add("block_mask", node.input(BlockSparseAttentionInputIdx::BLOCK_MASK).id());
    block_sparse_attention_info.add("block_size_q", desc->block_size_q);
    block_sparse_attention_info.add("block_size_kv", desc->block_size_kv);
    block_sparse_attention_info.add("causal", desc->causal);

    node_info->add("block_sparse_attention_info", block_sparse_attention_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

block_sparse_attention_inst::typed_primitive_inst(network& network, const block_sparse_attention_node& node) : parent(network, node) {}
}  // namespace cldnn
