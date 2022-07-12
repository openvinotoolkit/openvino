// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <read_value_inst.h>
#include "primitive_type_base.h"
#include <sstream>
#include <json_object.h>
#include <data_inst.h>

namespace cldnn {

primitive_type_id read_value::type_id() {
    static primitive_type_base<read_value> instance;
    return &instance;
}

read_value_inst::typed_primitive_inst(network& network, const read_value_node& node) :
    parent(network, node, false),
    memory_state::variable{node.get_primitive()->variable_id} {
}

layout read_value_inst::calc_output_layout(const read_value_node& node) {
    return node.get_primitive()->output_layout;
}

std::string read_value_inst::to_string(const read_value_node& node) {
    auto node_info = node.desc_to_json();

    json_composite read_value_info;
    read_value_info.add("input id", node.input().id());
    read_value_info.add("variable id", node.get_primitive()->variable_id);
    node_info->add("read_value info", read_value_info);

    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

} // namespace cldnn
