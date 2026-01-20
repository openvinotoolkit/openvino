// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_scatter_reduction_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include "to_string_utils.h"
#include <string>
#include <vector>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(moe_scatter_reduction)

layout moe_scatter_reduction_inst::calc_output_layout(moe_scatter_reduction_node const& node, kernel_impl_params const& impl_param) {
    auto output_layouts = calc_output_layouts<ov::PartialShape>(node, impl_param);
    return output_layouts[0];
}

template<typename ShapeType>
std::vector<layout> moe_scatter_reduction_inst::calc_output_layouts(moe_scatter_reduction_node const& /*node*/, const kernel_impl_params& impl_param) {
    const auto& desc = impl_param.typed_desc<moe_scatter_reduction>();
    const auto num_active_experts_per_token = desc->num_active_experts_per_token;

    const auto& input_shape = impl_param.input_layouts[0].get<ShapeType>();
    const auto& hidden_size = input_shape[input_shape.size() - 1];
    OPENVINO_ASSERT(hidden_size.is_static(), impl_param.desc->id, " hidden size dimension (shape[1]) must be static");
    if (impl_param.input_layouts[0].is_dynamic())
        return {layout{ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension(hidden_size)},
                impl_param.input_layouts[0].data_type, impl_param.input_layouts[0].format}};
    if (desc->has_batch_dim) {
        const auto num_tokens = impl_param.input_layouts[0].get_shape()[1] / num_active_experts_per_token;
        const auto& out_shape = ov::PartialShape{1, ov::Dimension(num_tokens), ov::Dimension(hidden_size)};
        return {layout{out_shape, impl_param.input_layouts[0].data_type, impl_param.input_layouts[0].format}};
    } else {
        const auto num_tokens = impl_param.input_layouts[0].get_shape()[0] / num_active_experts_per_token;
        const auto& out_shape = ov::PartialShape{ov::Dimension(num_tokens), 1, ov::Dimension(hidden_size)};
        return {layout{out_shape, impl_param.input_layouts[0].data_type, impl_param.input_layouts[0].format}};
    }
}

template std::vector<layout> moe_scatter_reduction_inst::calc_output_layouts<ov::PartialShape>(moe_scatter_reduction_node const& node,
    const kernel_impl_params& impl_param);

std::string moe_scatter_reduction_inst::to_string(moe_scatter_reduction_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite moe_scatter_reduction_info;
    moe_scatter_reduction_info.add("input id", node.input().id());
    if (desc->output_data_types[0].has_value())
       moe_scatter_reduction_info.add("out dt: ", dt_to_str(*desc->output_data_types[0]));
    node_info->add("moe_scatter_reduction_info", moe_scatter_reduction_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

moe_scatter_reduction_inst::typed_primitive_inst(network& network, moe_scatter_reduction_node const& node) : parent(network, node) { }
}  // namespace cldnn

