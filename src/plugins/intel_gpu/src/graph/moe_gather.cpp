// Copyright (C) 2025 Intel Corporation
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
    // TODO (not implemented yet)
    return impl_param.input_layouts[0];
}

template<typename ShapeType>
std::vector<layout> moe_gather_inst::calc_output_layouts(moe_gather_node const& /*node*/, const kernel_impl_params& impl_param) {
    const auto& desc = impl_param.typed_desc<moe_gather>();
    const auto num_experts_per_token = desc->num_experts_per_token;
    const auto hidden_size = impl_param.input_layouts[0].get_shape()[1];
    if (impl_param.input_layouts[0].is_dynamic())
        return {layout{ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension(hidden_size)},
                impl_param.input_layouts[0].data_type, impl_param.input_layouts[0].format}};
    const auto num_tokens = impl_param.input_layouts[0].get_shape()[0];
    const auto& out_shape = ov::PartialShape{ov::Dimension(num_tokens * num_experts_per_token), ov::Dimension(hidden_size)};
    return {layout{out_shape, impl_param.input_layouts[0].data_type, impl_param.input_layouts[0].format}};
}

template std::vector<layout> moe_gather_inst::calc_output_layouts<ov::PartialShape>(moe_gather_node const& node, const kernel_impl_params& impl_param);

std::string moe_gather_inst::to_string(moe_gather_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite moe_gather_info;
    if (desc->output_data_types[0].has_value())
        moe_gather_info.add("out dt: ", dt_to_str(*desc->output_data_types[0]));
    node_info->dump(primitive_description);

    return primitive_description.str();
}

moe_gather_inst::typed_primitive_inst(network& network, moe_gather_node const& node) : parent(network, node) { }
}  // namespace cldnn
