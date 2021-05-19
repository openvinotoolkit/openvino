// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_inst.h"

#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id gather::type_id() {
    static primitive_type_base<gather> instance;
    return &instance;
}

layout gather_inst::calc_output_layout(gather_node const& node) {
    auto desc = node.get_primitive();

    auto input_layout = node.input(0).get_output_layout();
    auto output_format = desc->output_format;
    auto output_shape = desc->output_shape;

    auto output_type = input_layout.data_type;
    if (node.has_fused_primitives()) {
        output_type = node.get_fused_output_layout().data_type;
    }

    return layout{output_type, output_format, output_shape};
}

std::string gather_inst::to_string(gather_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite gather_info;
    gather_info.add("input id", input.id());
    gather_info.add("axis", desc->axis);
    gather_info.add("batch_dim", desc->batch_dim);
    gather_info.add("output shape", desc->output_shape.to_string());

    node_info->add("gather info", gather_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

gather_inst::typed_primitive_inst(network_impl& network, gather_node const& node) : parent(network, node) {}

}  // namespace cldnn
