// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(mvn)

layout mvn_inst::calc_output_layout(const mvn_node& node, const kernel_impl_params& impl_param) {
    auto output_layouts = calc_output_layouts<ov::PartialShape>(node, impl_param);
    return output_layouts[0];
}

template <typename ShapeType>
std::vector<layout> mvn_inst::calc_output_layouts(const mvn_node& /*node*/, const kernel_impl_params& impl_param) {
    auto in_layout = impl_param.get_input_layout(0);
    auto output_type = impl_param.desc->output_data_types[0].value_or(in_layout.data_type);

    // Normalized output is fractional: an i8/u8 output type is only valid via a fused quantize.
    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_output_element_type();
    } else if (data_type_traits::is_i8_u8(in_layout.data_type)) {
        output_type = data_types::f32;
    }

    return {layout(in_layout.get<ShapeType>(), output_type, in_layout.format)};
}

template std::vector<layout> mvn_inst::calc_output_layouts<ov::PartialShape>(mvn_node const& node,
                                                                             const kernel_impl_params& impl_param);

std::string mvn_inst::to_string(mvn_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();
    auto epsilon = desc->epsilon;
    auto axes = desc->reduction_axes;
    const auto* normalize_variance = desc->normalize_variance ? "true" : "false";
    const auto* eps_inside_sqrt = desc->eps_inside_sqrt ? "true" : "false";
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite mvn_info;
    mvn_info.add("input id", input.id());
    mvn_info.add("epsilon", epsilon);
    mvn_info.add("reduction axes", std::move(axes));
    mvn_info.add("normalize_variance region", normalize_variance);
    mvn_info.add("eps_inside_sqrt region", eps_inside_sqrt);

    node_info->add("mvn info", mvn_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

mvn_inst::typed_primitive_inst(network& network, mvn_node const& node) : parent(network, node) {}
}  // namespace cldnn
