// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sstream>
#include <string>
#include <vector>

#include "json_object.h"
#include "primitive_type_base.h"
#include "qsa_indexer_inst.h"
#include "to_string_utils.h"

namespace cldnn {

GPU_DEFINE_PRIMITIVE_TYPE_ID(qsa_indexer)

layout qsa_indexer_inst::calc_output_layout(const qsa_indexer_node& node, const kernel_impl_params& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
std::vector<layout> qsa_indexer_inst::calc_output_layouts(const qsa_indexer_node& node, const kernel_impl_params& impl_param) {
    const auto& desc = impl_param.typed_desc<qsa_indexer>();
    const auto q_idx_shape = impl_param.get_input_layout(qsa_indexer::InputIdx::Q_IDX).get<ShapeType>();
    OPENVINO_ASSERT(q_idx_shape.rank().is_static() && q_idx_shape.size() == 4, "[GPU] qsa_indexer expects a rank-4 [B, L, H_iq, D_idx] q_idx input");

    const auto h_sel_dim = ov::Dimension(static_cast<int64_t>(desc->h_sel));
    ShapeType sel_indices_shape{q_idx_shape[0], h_sel_dim, q_idx_shape[1], ov::Dimension(static_cast<int64_t>(desc->block_topk))};
    ShapeType sel_count_shape{q_idx_shape[0], h_sel_dim, q_idx_shape[1]};

    std::vector<layout> layouts = {layout{sel_indices_shape, ov::element::i32, cldnn::format::get_default_format(sel_indices_shape.size())},
                                   layout{sel_count_shape, ov::element::i32, cldnn::format::get_default_format(sel_count_shape.size())}};
    // The updated state has the shape and type of the corresponding state input.
    for (size_t in : {qsa_indexer::InputIdx::SUMMARY_CACHE, qsa_indexer::InputIdx::PAST_INCOMPLETE_RAW_K, qsa_indexer::InputIdx::PAST_INCOMPLETE_ROPE}) {
        const auto& in_layout = impl_param.get_input_layout(in);
        const auto shape = in_layout.get<ShapeType>();
        layouts.emplace_back(shape, in_layout.data_type, cldnn::format::get_default_format(shape.size()));
    }
    return layouts;
}

template std::vector<layout> qsa_indexer_inst::calc_output_layouts<ov::PartialShape>(const qsa_indexer_node& node, const kernel_impl_params& impl_param);

std::string qsa_indexer_inst::to_string(const qsa_indexer_node& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;
    json_composite info;
    info.add("compress_ratio", desc->compress_ratio);
    info.add("block_topk", desc->block_topk);
    info.add("h_sel", desc->h_sel);
    info.add("indexer_head_dim", desc->indexer_head_dim);
    info.add("indexer_rotary_dim", desc->indexer_rotary_dim);
    info.add("scale", desc->scale);
    node_info->add("qsa_indexer_info", info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

qsa_indexer_inst::typed_primitive_inst(network& network, const qsa_indexer_node& node) : parent(network, node) {
    std::string reason;
    OPENVINO_ASSERT(ov::intel_gpu::op::QSAIndexer::validate_config(node.get_primitive()->get_config(), reason), "[GPU] Invalid qsa_indexer primitive: ", reason);
}

}  // namespace cldnn
