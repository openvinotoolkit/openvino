// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_gather_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include "to_string_utils.h"
#include <string>
#include <vector>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(moe_gather)

layout moe_gather_inst::calc_output_layout(moe_gather_node const& node, kernel_impl_params const& impl_param) {
    auto output_layouts = calc_output_layouts<ov::PartialShape>(node, impl_param);
    return output_layouts[0];
}

template <typename ShapeType>
std::vector<layout> moe_gather_inst::calc_output_layouts(const moe_gather_node& /*node*/, const kernel_impl_params& impl_param) {
    const auto& desc = impl_param.typed_desc<moe_gather>();
    const auto num_experts_per_token = desc->num_experts_per_token;
    const auto& in_layout = impl_param.input_layouts[0];
    const auto& input_shape = in_layout.get<ShapeType>();
    const auto& hidden_size = input_shape[input_shape.size() - 1];
    OPENVINO_ASSERT(hidden_size.is_static(), impl_param.desc->id, " hidden size dimension (shape[1]) must be static");

    if (in_layout.is_dynamic()) {
        return {layout{ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension(hidden_size)},
                in_layout.data_type, in_layout.format}};
    }
    const auto num_tokens = input_shape.size() == 2 ? input_shape[0] : desc->has_batch_dim ? input_shape[1] : input_shape[0];
    if (desc->has_batch_dim) {
        const auto& out_shape = ov::PartialShape{ov::Dimension(1), ov::Dimension(num_tokens * num_experts_per_token), ov::Dimension(hidden_size)};
        return {layout{out_shape, in_layout.data_type, in_layout.format}};
    } else {
        const auto& out_shape = ov::PartialShape{ov::Dimension(num_tokens * num_experts_per_token), ov::Dimension(1), ov::Dimension(hidden_size)};
        return {layout{out_shape, in_layout.data_type, in_layout.format}};
    }
}

template std::vector<layout> moe_gather_inst::calc_output_layouts<ov::PartialShape>(moe_gather_node const& node, const kernel_impl_params& impl_param);

std::string moe_gather_inst::to_string(moe_gather_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite moe_gather_info;
    if (desc->output_data_types[0].has_value())
        moe_gather_info.add("out dt: ", dt_to_str(*desc->output_data_types[0]));
    node_info->add("moe_gather info", moe_gather_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

moe_gather_inst::typed_primitive_inst(network& network, moe_gather_node const& node) : parent(network, node) { }
}  // namespace cldnn
