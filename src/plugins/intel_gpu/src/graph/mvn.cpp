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

layout mvn_inst::calc_output_layout(mvn_node const& node) {
    auto input_node_layout = node.input().get_non_padded_output_layout();
    auto output_type = node.get_primitive()->output_data_types.at(0) ? *node.get_primitive()->output_data_types.at(0) : input_node_layout.data_type;

    if (node.has_fused_primitives()) {
        output_type = node.get_fused_output_layout().data_type;
    } else if (input_node_layout.data_type == data_types::u8 || input_node_layout.data_type == data_types::i8) {
        output_type = data_types::f32;
    }

    return layout(output_type, input_node_layout.format, input_node_layout.size);
}

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
