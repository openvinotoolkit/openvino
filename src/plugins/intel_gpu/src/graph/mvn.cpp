// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id mvn::type_id() {
    static primitive_type_base<mvn> instance;
    return &instance;
}

layout mvn_inst::calc_output_layout(mvn_node const& node, kernel_impl_params const& impl_param) {
    auto input_node_layout = impl_param.get_non_padded_input_layout();
    auto output_type = impl_param.desc->output_data_type ? *impl_param.desc->output_data_type : input_node_layout.data_type;

    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_fused_output_layout().data_type;
    } else if (input_node_layout.data_type == data_types::u8 || input_node_layout.data_type == data_types::i8) {
        output_type = data_types::f32;
    }

    return layout(output_type, input_node_layout.format, input_node_layout.get_tensor());
}

template<typename ShapeType>
std::vector<layout> mvn_inst::calc_output_layouts(mvn_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<mvn>();
    auto input_layout = impl_param.get_input_layout(0);

    auto output_type = impl_param.desc->output_data_type ? *impl_param.desc->output_data_type
                                                         : input_layout.data_type;
    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_fused_output_layout().data_type;
    }

    ShapeType input_shape = input_layout.get<ShapeType>();
    ShapeType output_shape = input_shape;

    format output_format = format::adjust_to_rank(input_layout.format, output_shape.size());

    return { layout{output_shape, output_type, output_format} };
}

template std::vector<layout> mvn_inst::calc_output_layouts<ov::PartialShape>(mvn_node const& node, const kernel_impl_params& impl_param);

std::string mvn_inst::to_string(mvn_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();
    auto epsilon = desc->epsilon;
    auto across_channels = desc->across_channels ? "true" : "false";
    auto normalize_variance = desc->normalize_variance ? "true" : "false";
    auto eps_inside_sqrt = desc->eps_inside_sqrt ? "true" : "false";
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite mvn_info;
    mvn_info.add("input id", input.id());
    mvn_info.add("epsilon", epsilon);
    mvn_info.add("across_channels region", across_channels);
    mvn_info.add("normalize_variance region", normalize_variance);
    mvn_info.add("eps_inside_sqrt region", eps_inside_sqrt);

    node_info->add("mvn info", mvn_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

mvn_inst::typed_primitive_inst(network& network, mvn_node const& node) : parent(network, node) {}
}  // namespace cldnn
