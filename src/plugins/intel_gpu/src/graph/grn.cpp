// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "grn_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id grn::type_id() {
    static primitive_type_base<grn> instance;
    return &instance;
}

layout grn_inst::calc_output_layout(grn_node const& node) {
    auto input_node_layout = node.input().get_non_padded_output_layout();
    auto output_type = node.get_primitive()->output_data_type ? *node.get_primitive()->output_data_type : input_node_layout.data_type;

    return layout(output_type, input_node_layout.format, input_node_layout.size);
}

std::string grn_inst::to_string(grn_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();
    auto bias = desc->bias;
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite grn_info;
    grn_info.add("input id", input.id());
    grn_info.add("bias", bias);

    node_info->add("grn info", grn_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

grn_inst::typed_primitive_inst(network& network, grn_node const& node) : parent(network, node) {}
}  // namespace cldnn
