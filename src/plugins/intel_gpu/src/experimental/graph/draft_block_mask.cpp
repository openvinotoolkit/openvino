// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sstream>
#include <string>
#include <vector>

#include "draft_block_mask_inst.h"
#include "json_object.h"
#include "primitive_type_base.h"
#include "to_string_utils.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(draft_block_mask)

layout draft_block_mask_inst::calc_output_layout(const draft_block_mask_node& node, const kernel_impl_params& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
std::vector<layout> draft_block_mask_inst::calc_output_layouts(const draft_block_mask_node& /*node*/, const kernel_impl_params& impl_param) {
    OPENVINO_ASSERT(impl_param.input_layouts.size() == 3 || impl_param.input_layouts.size() == 5, "draft_block_mask must have 3 or 5 inputs");
    const auto desc = impl_param.typed_desc<draft_block_mask>();

    const auto& q_shape = impl_param.get_input_layout(DraftBlockMaskInputIdx::Q_WINDOWS).get_partial_shape();
    const auto& k_shape = impl_param.get_input_layout(DraftBlockMaskInputIdx::K_WINDOWS).get_partial_shape();

    const auto Nq = q_shape.rank().is_static() ? q_shape[0] : ov::Dimension::dynamic();
    const auto Nk = k_shape.rank().is_static() ? k_shape[0] : ov::Dimension::dynamic();

    const ov::PartialShape output_shape{ov::Dimension(desc->batch_size), ov::Dimension(desc->num_heads), Nq, Nk};
    return {layout(output_shape, data_types::u8, format::bfyx)};
}

template std::vector<layout> draft_block_mask_inst::calc_output_layouts<ov::PartialShape>(const draft_block_mask_node& node,
                                                                                          const kernel_impl_params& impl_param);

std::string draft_block_mask_inst::to_string(const draft_block_mask_node& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite draft_block_mask_info;
    draft_block_mask_info.add("q_windows", node.input(DraftBlockMaskInputIdx::Q_WINDOWS).id());
    draft_block_mask_info.add("k_windows", node.input(DraftBlockMaskInputIdx::K_WINDOWS).id());
    draft_block_mask_info.add("local_mask", node.input(DraftBlockMaskInputIdx::LOCAL_MASK).id());
    draft_block_mask_info.add("num_heads", desc->num_heads);
    if (node.get_dependencies().size() == 5) {
        draft_block_mask_info.add("seqlen", node.input(DraftBlockMaskInputIdx::SEQLEN).id());
        draft_block_mask_info.add("topk", node.input(DraftBlockMaskInputIdx::TOPK).id());
    } else {
        draft_block_mask_info.add("seqlen", desc->seqlen);
        draft_block_mask_info.add("topk", desc->topk);
    }
    draft_block_mask_info.add("batch_size", desc->batch_size);

    node_info->add("draft_block_mask_info", draft_block_mask_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

draft_block_mask_inst::typed_primitive_inst(network& network, const draft_block_mask_node& node) : parent(network, node) {}
}  // namespace cldnn
