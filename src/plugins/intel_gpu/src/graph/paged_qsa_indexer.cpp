// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sstream>
#include <string>
#include <vector>

#include "json_object.h"
#include "paged_qsa_indexer_inst.h"
#include "primitive_type_base.h"
#include "to_string_utils.h"

namespace cldnn {

GPU_DEFINE_PRIMITIVE_TYPE_ID(paged_qsa_indexer)

layout paged_qsa_indexer_inst::calc_output_layout(const paged_qsa_indexer_node& node, const kernel_impl_params& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
std::vector<layout> paged_qsa_indexer_inst::calc_output_layouts(const paged_qsa_indexer_node& node, const kernel_impl_params& impl_param) {
    const auto& desc = impl_param.typed_desc<paged_qsa_indexer>();
    const auto index_query_shape = impl_param.get_input_layout(paged_qsa_indexer::InputIdx::INDEX_QUERY).get<ShapeType>();
    OPENVINO_ASSERT(index_query_shape.rank().is_static() && index_query_shape.size() == 2,
                    "[GPU] paged_qsa_indexer expects a rank-2 [tokens, H_iq*D_idx] index_query input");

    const auto num_selection_heads_dim = ov::Dimension(static_cast<int64_t>(desc->num_selection_heads));
    ShapeType selected_block_indices_shape{index_query_shape[0], num_selection_heads_dim, ov::Dimension(static_cast<int64_t>(desc->block_topk))};
    ShapeType selected_block_count_shape{index_query_shape[0], num_selection_heads_dim};

    return {layout{selected_block_indices_shape, ov::element::i32, cldnn::format::get_default_format(selected_block_indices_shape.size())},
            layout{selected_block_count_shape, ov::element::i32, cldnn::format::get_default_format(selected_block_count_shape.size())}};
}

template std::vector<layout> paged_qsa_indexer_inst::calc_output_layouts<ov::PartialShape>(const paged_qsa_indexer_node& node,
                                                                                           const kernel_impl_params& impl_param);

std::string paged_qsa_indexer_inst::to_string(const paged_qsa_indexer_node& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;
    json_composite info;
    info.add("key_block_size", desc->key_block_size);
    info.add("block_topk", desc->block_topk);
    info.add("num_selection_heads", desc->num_selection_heads);
    info.add("indexer_head_dim", desc->indexer_head_dim);
    info.add("indexer_rotary_dim", desc->indexer_rotary_dim);
    info.add("scale", desc->scale);
    node_info->add("paged_qsa_indexer_info", info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

paged_qsa_indexer_inst::typed_primitive_inst(network& network, const paged_qsa_indexer_node& node) : parent(network, node) {
    std::string reason;
    OPENVINO_ASSERT(ov::intel_gpu::op::PagedQSAIndexer::validate_config(node.get_primitive()->get_config(), reason),
                    "[GPU] Invalid paged_qsa_indexer primitive: ",
                    reason);
}

}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::paged_qsa_indexer)
